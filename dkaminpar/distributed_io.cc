/*******************************************************************************
 * @file:   distributed_io.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Load / store distributed graphs from METIS or KaHIP Binary formats.
 ******************************************************************************/
#include "dkaminpar/distributed_io.h"

#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"
#include "dkaminpar/utils/math.h"
#include "kaminpar/io.h"
#include "kaminpar/utils/strings.h"

namespace dkaminpar::io {
SET_DEBUG(false);

DistributedGraph read_graph(const std::string& filename, DistributionType type, MPI_Comm comm) {
    if (shm::utility::str::ends_with(filename, "bgf") || shm::utility::str::ends_with(filename, "bin")) {
        if (type == DistributionType::NODE_BALANCED) {
            return binary::read_node_balanced(filename, comm);
        } else {
            return binary::read_edge_balanced(filename, comm);
        }
    }

    if (type == DistributionType::NODE_BALANCED) {
        return metis::read_node_balanced(filename, comm);
    } else {
        return metis::read_edge_balanced(filename, comm);
    }
}

namespace metis {
namespace shm = kaminpar::io::metis;

DistributedGraph read_node_balanced(const std::string& filename, MPI_Comm comm) {
    const auto comm_info = mpi::get_comm_info();
    const PEID size      = comm_info.first;
    const PEID rank      = comm_info.second;

    graph::Builder builder{comm};

    GlobalNodeID current = 0;
    GlobalNodeID from    = 0;
    GlobalNodeID to      = 0;

    shm::read_observable(
        filename,
        [&](const auto& format) {
            const auto                  global_n = static_cast<GlobalNodeID>(format.number_of_nodes);
            [[maybe_unused]] const auto global_m = static_cast<GlobalNodeID>(format.number_of_edges) * 2;
            DBG << "Loading graph with global_n=" << global_n << " and global_m=" << global_m;

            scalable_vector<GlobalNodeID> node_distribution(size + 1);
            for (PEID p = 0; p < size; ++p) {
                const auto [p_from, p_to] = math::compute_local_range<GlobalNodeID>(global_n, size, p);
                node_distribution[p + 1]  = p_to;
            }
            ASSERT(node_distribution.front() == 0);
            ASSERT(node_distribution.back() == global_n);

            from = node_distribution[rank];
            to   = node_distribution[rank + 1];
            DBG << "PE " << rank << ": from=" << from << " to=" << to << " n=" << format.number_of_nodes;

            builder.initialize(std::move(node_distribution));
        },
        [&](const std::uint64_t& u_weight) {
            ++current;
            if (current > to) {
                return false;
            }
            if (current > from) {
                builder.create_node(static_cast<NodeWeight>(u_weight));
            }
            return true;
        },
        [&](const std::uint64_t& e_weight, const std::uint64_t& v) {
            if (current > from) {
                builder.create_edge(static_cast<EdgeWeight>(e_weight), static_cast<GlobalNodeID>(v));
            }
        });

    return builder.finalize();
}

DistributedGraph read_edge_balanced(const std::string& filename, MPI_Comm comm) {
    const auto comm_info = mpi::get_comm_info();
    const PEID size      = comm_info.first;
    const PEID rank      = comm_info.second;

    PEID         current_pe   = 0;
    GlobalNodeID current_node = 0;
    GlobalEdgeID current_edge = 0;
    GlobalEdgeID to           = 0;

    scalable_vector<EdgeID>                  nodes;
    scalable_vector<GlobalNodeID>            global_edges;
    scalable_vector<NodeWeight>              node_weights;
    scalable_vector<EdgeWeight>              edge_weights;
    scalable_vector<PEID>                    ghost_owner;
    scalable_vector<GlobalNodeID>            ghost_to_global;
    std::unordered_map<GlobalNodeID, NodeID> global_to_ghost;

    scalable_vector<GlobalNodeID> node_distribution(size + 1);
    scalable_vector<GlobalEdgeID> edge_distribution(size + 1);

    // read graph file
    shm::read_observable(
        filename,
        [&](const auto& format) {
            const auto global_n         = static_cast<GlobalNodeID>(format.number_of_nodes);
            const auto global_m         = static_cast<GlobalEdgeID>(format.number_of_edges) * 2;
            node_distribution.back()    = global_n;
            edge_distribution.back()    = global_m;
            const auto [pe_from, pe_to] = math::compute_local_range<GlobalEdgeID>(global_m, size, current_pe);
            to                          = pe_to;
        },
        [&](const std::uint64_t& u_weight) {
            if (current_edge >= to) {
                node_distribution[current_pe] = current_node;
                edge_distribution[current_pe] = current_edge;
                ++current_pe;

                const GlobalEdgeID global_m = edge_distribution.back();
                const auto [pe_from, pe_to] = math::compute_local_range<GlobalEdgeID>(global_m, size, current_pe);
                to                          = pe_to;
            }

            if (current_pe == rank) {
                nodes.push_back(global_edges.size());
                node_weights.push_back(static_cast<NodeWeight>(u_weight));
            }

            ++current_node;
            return true;
        },
        [&](const std::uint64_t& e_weight, const std::uint64_t& v) {
            if (current_pe == rank) {
                global_edges.push_back(static_cast<GlobalNodeID>(v));
                edge_weights.push_back(static_cast<EdgeWeight>(e_weight));
            }
            ++current_edge;
        });

    // at this point we should have a valid node and edge distribution
    const GlobalNodeID offset_n = node_distribution[rank];
    const auto         local_n  = static_cast<NodeID>(node_distribution[rank + 1] - node_distribution[rank]);

    // remap global edges to local edges and create ghost PEs
    scalable_vector<NodeID> edges(global_edges.size());
    for (std::size_t i = 0; i < global_edges.size(); ++i) {
        const GlobalNodeID global_v = global_edges[i];
        if (offset_n <= global_v && global_v < offset_n + local_n) { // owned node
            edges[i] = static_cast<NodeID>(global_v - offset_n);
        } else { // ghost node
            if (!global_to_ghost.contains(global_v)) {
                const NodeID local_id = local_n + ghost_to_global.size();
                ghost_to_global.push_back(global_v);
                global_to_ghost[global_v] = local_id;

                auto       it    = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), global_v);
                const auto owner = static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
                ghost_owner.push_back(owner);
            }

            edges[i] = global_to_ghost[global_v];
        }
    }

    // init graph
    return {
        std::move(node_distribution),
        std::move(edge_distribution),
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights),
        std::move(ghost_owner),
        std::move(ghost_to_global),
        std::move(global_to_ghost),
        comm};
}

void write(
    const std::string& filename, const DistributedGraph& graph, const bool write_node_weights,
    const bool write_edge_weights) {
    if (mpi::get_comm_rank() == 0) { // clear file
        std::ofstream tmp(filename);
    }
    mpi::barrier();

    mpi::sequentially([&](const PEID pe) {
        std::ofstream out(filename, std::ios_base::out | std::ios_base::app);
        if (pe == 0) {
            out << graph.global_n() << " " << graph.global_m() / 2;
            if (write_node_weights || write_edge_weights) {
                out << " ";
                out << static_cast<int>(write_node_weights);
                out << static_cast<int>(write_edge_weights);
            }
            out << "\n";
        }

        for (const NodeID u: graph.nodes()) {
            if (write_node_weights) {
                out << graph.node_weight(u) << " ";
            }
            for (const auto [e, v]: graph.neighbors(u)) {
                out << graph.local_to_global_node(v) + 1 << " ";
                if (write_edge_weights) {
                    out << graph.edge_weight(e) << " ";
                }
            }
            out << "\n";
        }
    });
}
} // namespace metis

namespace binary {
using IDType = unsigned long long;

namespace {
std::pair<IDType, IDType> read_header(std::ifstream& in) {
    IDType version, global_n, global_m;
    in.read(reinterpret_cast<char*>(&version), sizeof(IDType));
    ALWAYS_ASSERT(version == 3) << "invalid binary graph format!";

    in.read(reinterpret_cast<char*>(&global_n), sizeof(IDType));
    in.read(reinterpret_cast<char*>(&global_m), sizeof(IDType));

    return {global_n, global_m};
}

DistributedGraph
read_distributed_graph(std::ifstream& in, const GlobalNodeID from, const GlobalNodeID to, MPI_Comm comm) {
    const auto n = static_cast<NodeID>(to - from);

    // read nodes
    scalable_vector<EdgeID> nodes(n + 1);
    IDType                  first_edge_index         = 0;
    IDType                  first_invalid_edge_index = 0;
    {
        // read part of global nodes array
        scalable_vector<IDType> global_nodes(n + 1);
        const std::streamsize   offset = 3 * sizeof(IDType) + from * sizeof(IDType);
        const std::streamsize   length = (n + 1) * sizeof(IDType);
        in.seekg(offset);
        in.read(reinterpret_cast<char*>(global_nodes.data()), length);

        // build local nodes array
        first_edge_index         = global_nodes.front();
        first_invalid_edge_index = global_nodes.back();

        tbb::parallel_for<std::size_t>(0, global_nodes.size(), [&](const std::size_t i) {
            nodes[i] = static_cast<EdgeID>((global_nodes[i] - first_edge_index) / sizeof(IDType));
        });
    }
    const EdgeID m = nodes.back();

    // read edges
    scalable_vector<NodeID> edges(m);

    // read part of global edge array
    scalable_vector<IDType> global_edges(m);
    const std::streamsize   offset = first_edge_index;
    const std::streamsize   length = first_invalid_edge_index - first_edge_index;
    in.seekg(offset);
    in.read(reinterpret_cast<char*>(global_edges.data()), length);

    auto node_distribution = mpi::build_distribution_from_local_count<GlobalNodeID, scalable_vector>(n, comm);
    auto edge_distribution = mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(m, comm);

    // map ghost nodes to local nodes
    graph::GhostNodeMapper mapper(node_distribution, comm);
    tbb::parallel_for<std::size_t>(0, global_edges.size(), [&](const std::size_t i) {
        const GlobalNodeID edge_target = global_edges[i];
        if (edge_target < from || edge_target >= to) {
            mapper.new_ghost_node(edge_target);
        }
    });
    auto  ghost_mapping_result = mapper.finalize();
    auto& global_to_ghost      = ghost_mapping_result.global_to_ghost;
    auto& ghost_to_global      = ghost_mapping_result.ghost_to_global;
    auto& ghost_owner          = ghost_mapping_result.ghost_owner;

    // map edges to local edges
    tbb::parallel_for<std::size_t>(0, global_edges.size(), [&](const std::size_t i) {
        const GlobalNodeID edge_target = global_edges[i];
        if (from <= edge_target && edge_target < to) {
            edges[i] = static_cast<NodeID>(edge_target - from);
        } else {
            edges[i] = (*global_to_ghost.find(edge_target + 1)).second;
        }
    });

    return {std::move(node_distribution), std::move(edge_distribution), std::move(nodes),           std::move(edges),
            std::move(ghost_owner),       std::move(ghost_to_global),   std::move(global_to_ghost), comm};
}
} // namespace

DistributedGraph read_node_balanced(const std::string& filename, MPI_Comm comm) {
    std::ifstream in(filename);

    const auto [global_n, global_m] = read_header(in);

    const auto [size, rank] = mpi::get_comm_info(comm);
    const auto  local_range = math::compute_local_range<GlobalNodeID>(global_n, size, rank);
    const auto& from        = local_range.first;
    const auto& to          = local_range.second;

    return read_distributed_graph(in, static_cast<GlobalNodeID>(from), static_cast<GlobalNodeID>(to), comm);
}

namespace {
IDType adj_list_offset_to_edge(const IDType n, const IDType offset) {
    return (offset / sizeof(IDType)) - 3 - (n + 1);
}

IDType read_first_edge(std::ifstream& in, const IDType n, const IDType u) {
    ASSERT(u < n);

    const IDType offset = (3 + u) * sizeof(IDType);
    in.seekg(static_cast<std::streamsize>(offset));

    IDType entry = 0;
    in.read(reinterpret_cast<char*>(&entry), sizeof(IDType));
    return adj_list_offset_to_edge(n, entry);
}

IDType read_first_invalid_edge(std::ifstream& in, const IDType n, const IDType u) {
    return read_first_edge(in, n, u + 1);
}

IDType
compute_edge_balanced_from_node(std::ifstream& in, const IDType n, const IDType m, const int rank, const int size) {
    if (rank == 0) {
        return 0;
    } else if (rank == size) {
        return n;
    }

    const IDType chunk     = m / size;
    const IDType remainder = m % size;
    const IDType target    = rank * chunk + std::min<IDType>(rank, remainder);

    std::pair<IDType, IDType> a{0, 0};
    std::pair<IDType, IDType> b{n - 1, m - 1};

    while (b.first - a.first > 1) {
        std::pair<IDType, IDType> mid;
        mid.first  = (a.first + b.first) / 2;
        mid.second = read_first_edge(in, n, mid.first);

        if (mid.second < target) {
            a = mid;
        } else {
            b = mid;
        }

        ASSERT(b.first >= a.first);
    }

    ASSERT(a.second <= target && target <= b.second);
    ASSERT(b.first < n);
    return b.first;
}

IDType
compute_edge_balanced_to_node(std::ifstream& in, const IDType n, const IDType m, const int rank, const int size) {
    return compute_edge_balanced_from_node(in, n, m, rank + 1, size);
}
} // namespace

DistributedGraph read_edge_balanced(const std::string& filename, MPI_Comm comm) {
    std::ifstream in(filename);

    const auto [global_n, global_m] = read_header(in);
    const auto [size, rank]         = mpi::get_comm_info(comm);
    const auto from                 = compute_edge_balanced_from_node(in, global_n, global_m, rank, size);
    const auto to                   = compute_edge_balanced_to_node(in, global_n, global_m, rank, size);
    SLOG << V(from) << V(to) << V(global_n) << V(global_m);

    return read_distributed_graph(in, static_cast<GlobalNodeID>(from), static_cast<GlobalNodeID>(to), comm);
}
} // namespace binary
} // namespace dkaminpar::io