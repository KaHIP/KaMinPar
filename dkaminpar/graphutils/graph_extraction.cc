/*******************************************************************************
 * @file:   graph_extraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#include "dkaminpar/graphutils/graph_extraction.h"

#include <algorithm>
#include <functional>

#include <mpi.h>

#include "common/parallel/algorithm.h"
#include "common/parallel/vector_ets.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/graphutils/graph_synchronization.h"
#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/graph_communication.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/utils/math.h"

namespace dkaminpar::graph {
SET_DEBUG(true);

namespace {
PEID compute_block_owner(const BlockID b, const BlockID k, const PEID num_pes) {
    return static_cast<PEID>(math::compute_local_range_rank<BlockID>(k, static_cast<BlockID>(num_pes), b));
}

auto count_block_induced_subgraph_sizes(const DistributedPartitionedGraph& p_graph) {
    parallel::vector_ets<NodeID> num_nodes_per_block_ets(p_graph.k());
    parallel::vector_ets<EdgeID> num_edges_per_block_ets(p_graph.k());

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto r) {
        auto& num_nodes_per_block = num_nodes_per_block_ets.local();
        auto& num_edges_per_block = num_edges_per_block_ets.local();
        for (NodeID u = r.begin(); u != r.end(); ++u) {
            const BlockID u_block = p_graph.block(u);
            ++num_nodes_per_block[u_block];
            for (const auto [e, v]: p_graph.neighbors(u)) {
                if (u_block == p_graph.block(v)) {
                    ++num_edges_per_block[u_block];
                }
            }
        }
    });

    return std::make_pair(num_nodes_per_block_ets.combine(std::plus{}), num_edges_per_block_ets.combine(std::plus{}));
}
} // namespace

// Build a local block-induced subgraph for each block of the graph partition.
ExtractedLocalSubgraphs extract_local_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph) {
    auto [num_nodes_per_block, num_edges_per_block] = count_block_induced_subgraph_sizes(p_graph);
    const EdgeID num_internal_edges = std::accumulate(num_edges_per_block.begin(), num_edges_per_block.end(), 0);

    ExtractedLocalSubgraphs memory;
    auto&                   shared_nodes          = memory.shared_nodes;
    auto&                   shared_node_weights   = memory.shared_node_weights;
    auto&                   shared_edges          = memory.shared_edges;
    auto&                   shared_edge_weights   = memory.shared_edge_weights;
    auto&                   nodes_offset          = memory.nodes_offset;
    auto&                   edges_offset          = memory.edges_offset;
    auto&                   mapping               = memory.mapping;
    auto                    next_node_in_subgraph = std::vector<parallel::Atomic<NodeID>>();

    // Allocate memory @todo
    {
        SCOPED_TIMER("Allocation", TIMER_DETAIL);

        const std::size_t min_nodes_size   = p_graph.n();
        const std::size_t min_edges_size   = num_internal_edges;
        const std::size_t min_offset_size  = p_graph.k() + 1;
        const std::size_t min_mapping_size = p_graph.total_n();

        KASSERT(shared_nodes.size() == shared_node_weights.size());
        KASSERT(shared_edges.size() == shared_edge_weights.size());
        KASSERT(nodes_offset.size() == edges_offset.size());

        if (shared_nodes.size() < min_nodes_size) {
            shared_nodes.resize(min_nodes_size);
            shared_node_weights.resize(min_nodes_size);
        }
        if (shared_edges.size() < min_edges_size) {
            shared_edges.resize(min_edges_size);
            shared_edge_weights.resize(min_edges_size);
        }
        if (nodes_offset.size() < min_offset_size) {
            nodes_offset.resize(min_offset_size);
            edges_offset.resize(min_offset_size);
        }
        if (mapping.size() < min_mapping_size) {
            mapping.resize(min_mapping_size);
        }

        next_node_in_subgraph.resize(p_graph.k());
    }

    // Compute of graphs in shared_* arrays
    {
        SCOPED_TIMER("Compute subgraph offsets", TIMER_DETAIL);

        parallel::prefix_sum(num_nodes_per_block.begin(), num_nodes_per_block.end(), nodes_offset.begin() + 1);
        parallel::prefix_sum(num_edges_per_block.begin(), num_edges_per_block.end(), edges_offset.begin() + 1);
    }

    // Compute node ID offset of local subgraph in global subgraphs
    std::vector<NodeID> global_node_offset(p_graph.k());
    mpi::exscan(num_nodes_per_block.data(), global_node_offset.data(), p_graph.k(), MPI_SUM, p_graph.communicator());

    // Build mapping from node IDs in p_graph to node IDs in the extracted subgraph
    {
        SCOPED_TIMER("Build node mapping", TIMER_DETAIL);

        // @todo bottleneck for scalability
        p_graph.pfor_nodes([&](const NodeID u) {
            const BlockID b               = p_graph.block(u);
            const NodeID  pos_in_subgraph = next_node_in_subgraph[b]++;
            const NodeID  pos             = nodes_offset[b] + pos_in_subgraph;
            shared_nodes[pos]             = u;
            mapping[u]                    = global_node_offset[b] + pos_in_subgraph;
        });
    }

    // Build mapping from local extract subgraph to global extracted subgraph for ghost nodes
    std::vector<NodeID> global_ghost_node_mapping(p_graph.ghost_n());

    {
        SCOPED_TIMER("Exchange ghost node mapping", TIMER_DETAIL);

        struct NodeToMappedNode {
            GlobalNodeID global_node;
            NodeID       mapped_node;
        };

        mpi::graph::sparse_alltoall_interface_to_pe<NodeToMappedNode>(
            p_graph.graph(),
            [&](const NodeID u) {
                return NodeToMappedNode{p_graph.local_to_global_node(u), mapping[u]};
            },
            [&](const auto buffer) {
                tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                    const auto& [global_node, mapped_node] = buffer[i];
                    const NodeID local_node                = p_graph.global_to_local_node(global_node);
                    mapping[local_node]                    = mapped_node;
                });
            });
    }

    // Extract the subgraphs
    {
        SCOPED_TIMER("Extract subgraphs", TIMER_DETAIL);

        tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
            const NodeID n0 = nodes_offset[b];
            const EdgeID e0 = edges_offset[b];
            EdgeID       e  = 0;

            // u, v, e = IDs in extracted subgraph
            // u_prime, v_prime, e_prime = IDs in p_graph
            for (NodeID u = 0; u < next_node_in_subgraph[b]; ++u) {
                const NodeID pos     = n0 + u;
                const NodeID u_prime = shared_nodes[pos];

                for (const auto [e_prime, v_prime]: p_graph.neighbors(u_prime)) {
                    if (p_graph.block(v_prime) != b) {
                        continue;
                    }

                    shared_edge_weights[e0 + e] = p_graph.edge_weight(e_prime);
                    shared_edges[e0 + e]        = mapping[v_prime];
                    ++e;
                }

                shared_nodes[pos]        = e;
                shared_node_weights[pos] = p_graph.node_weight(u_prime);
            }
        });
    }

    // remove global node offset in mapping -- we need the PE-relative value when copying the subgraph partitions pack
    // to the original graph
    p_graph.pfor_nodes([&](const NodeID u) {
        const BlockID b = p_graph.block(u);
        mapping[u] -= global_node_offset[b];
    });

    return memory;
}

namespace {
std::pair<std::vector<shm::Graph>, std::vector<std::vector<NodeID>>>
gather_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph, const ExtractedLocalSubgraphs& memory) {
    const PEID size = mpi::get_comm_size(p_graph.communicator());
    KASSERT(p_graph.k() % size == 0u, "k must be a multiple of #PEs", assert::always);
    const BlockID blocks_per_pe = p_graph.k() / size;

    // Communicate recvcounts
    struct GraphSize {
        NodeID n;
        EdgeID m;

        GraphSize operator+(const GraphSize other) {
            return {n + other.n, m + other.m};
        }

        GraphSize& operator+=(const GraphSize other) {
            n += other.n;
            m += other.m;
            return *this;
        }
    };

    std::vector<GraphSize> recv_subgraph_sizes(p_graph.k());
    {
        SCOPED_TIMER("Alltoall recvcounts", TIMER_DETAIL);

        START_TIMER("Compute counts", TIMER_DETAIL);
        std::vector<GraphSize> send_subgraph_sizes(p_graph.k());
        p_graph.pfor_blocks([&](const BlockID b) {
            send_subgraph_sizes[b].n = memory.nodes_offset[b + 1] - memory.nodes_offset[b];
            send_subgraph_sizes[b].m = memory.edges_offset[b + 1] - memory.edges_offset[b];
        });
        STOP_TIMER(TIMER_DETAIL);

        START_TIMER("MPI_Alltoall", TIMER_DETAIL);
        mpi::alltoall(
            send_subgraph_sizes.data(), blocks_per_pe, recv_subgraph_sizes.data(), blocks_per_pe,
            p_graph.communicator());
        STOP_TIMER(TIMER_DETAIL);
    }
    std::vector<GraphSize> recv_subgraph_displs(p_graph.k() + 1);
    parallel::prefix_sum(recv_subgraph_sizes.begin(), recv_subgraph_sizes.end(), recv_subgraph_displs.begin() + 1);

    std::vector<EdgeID>     shared_nodes;
    std::vector<NodeWeight> shared_node_weights;
    std::vector<NodeID>     shared_edges;
    std::vector<EdgeWeight> shared_edge_weights;

    {
        SCOPED_TIMER("Alltoallv block-induced subgraphs", TIMER_DETAIL);

        START_TIMER("Allocation", TIMER_DETAIL);
        std::vector<int> sendcounts_nodes(size);
        std::vector<int> sendcounts_edges(size);
        std::vector<int> sdispls_nodes(size + 1);
        std::vector<int> sdispls_edges(size + 1);
        std::vector<int> recvcounts_nodes(size);
        std::vector<int> recvcounts_edges(size);
        std::vector<int> rdispls_nodes(size + 1);
        std::vector<int> rdispls_edges(size + 1);
        STOP_TIMER(TIMER_DETAIL);

        START_TIMER("Compute counts and displs", TIMER_DETAIL);
        tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
            const BlockID first_block_on_pe         = pe * blocks_per_pe;
            const BlockID first_invalid_block_on_pe = (pe + 1) * blocks_per_pe;

            sendcounts_nodes[pe] =
                memory.nodes_offset[first_invalid_block_on_pe] - memory.nodes_offset[first_block_on_pe];
            sendcounts_edges[pe] =
                memory.edges_offset[first_invalid_block_on_pe] - memory.edges_offset[first_block_on_pe];

            for (BlockID b = first_block_on_pe; b < first_invalid_block_on_pe; ++b) {
                recvcounts_nodes[pe] += recv_subgraph_sizes[b].n;
                recvcounts_edges[pe] += recv_subgraph_sizes[b].m;
            }
        });
        parallel::prefix_sum(sendcounts_nodes.begin(), sendcounts_nodes.end(), sdispls_nodes.begin() + 1);
        parallel::prefix_sum(sendcounts_edges.begin(), sendcounts_edges.end(), sdispls_edges.begin() + 1);
        parallel::prefix_sum(recvcounts_nodes.begin(), recvcounts_nodes.end(), rdispls_nodes.begin() + 1);
        parallel::prefix_sum(recvcounts_edges.begin(), recvcounts_edges.end(), rdispls_edges.begin() + 1);
        STOP_TIMER(TIMER_DETAIL);

        shared_nodes.resize(rdispls_nodes.back());
        shared_node_weights.resize(rdispls_nodes.back());
        shared_edges.resize(rdispls_edges.back());
        shared_edge_weights.resize(rdispls_edges.back());

        START_TIMER("MPI_Alltoallv", TIMER_DETAIL);
        mpi::alltoallv(
            memory.shared_nodes.data(), sendcounts_nodes.data(), sdispls_nodes.data(), shared_nodes.data(),
            recvcounts_nodes.data(), rdispls_nodes.data(), p_graph.communicator());
        mpi::alltoallv(
            memory.shared_node_weights.data(), sendcounts_nodes.data(), sdispls_nodes.data(),
            shared_node_weights.data(), recvcounts_nodes.data(), rdispls_nodes.data(), p_graph.communicator());
        mpi::alltoallv(
            memory.shared_edges.data(), sendcounts_edges.data(), sdispls_edges.data(), shared_edges.data(),
            recvcounts_edges.data(), rdispls_edges.data(), p_graph.communicator());
        mpi::alltoallv(
            memory.shared_edge_weights.data(), sendcounts_edges.data(), sdispls_edges.data(),
            shared_edge_weights.data(), recvcounts_edges.data(), rdispls_edges.data(), p_graph.communicator());
        STOP_TIMER(TIMER_DETAIL);
    }

    std::vector<shm::Graph>          subgraphs(blocks_per_pe);
    std::vector<std::vector<NodeID>> offsets(blocks_per_pe);

    {
        SCOPED_TIMER("Construct subgraphs", TIMER_DETAIL);

        tbb::parallel_for<BlockID>(0, blocks_per_pe, [&](const BlockID b) {
            NodeID n = 0;
            EdgeID m = 0;
            for (PEID pe = 0; pe < size; ++pe) {
                const std::size_t i = b + pe * blocks_per_pe;
                n += recv_subgraph_sizes[i].n;
                m += recv_subgraph_sizes[i].m;
            }

            // Allocate memory for subgraph
            shm::StaticArray<EdgeID>     subgraph_nodes(n + 1);
            shm::StaticArray<NodeWeight> subgraph_node_weights(n);
            shm::StaticArray<NodeID>     subgraph_edges(m);
            shm::StaticArray<EdgeWeight> subgraph_edge_weights(m);

            // Copy subgraph to memory
            // @todo better approach might be to compute a prefix sum on recv_subgraph_sizes
            NodeID pos_n = 0;
            EdgeID pos_m = 0;

            for (PEID pe = 0; pe < size; ++pe) {
                const std::size_t id                    = pe * blocks_per_pe + b;
                const auto [num_nodes, num_edges]       = recv_subgraph_sizes[id];
                const auto [offset_nodes, offset_edges] = recv_subgraph_displs[id];
                offsets[b].push_back(pos_n);

                std::copy(
                    shared_nodes.begin() + offset_nodes, shared_nodes.begin() + offset_nodes + num_nodes,
                    subgraph_nodes.begin() + pos_n + 1);
                std::copy(
                    shared_node_weights.begin() + offset_nodes, shared_node_weights.begin() + offset_nodes + num_nodes,
                    subgraph_node_weights.begin() + pos_n);
                std::copy(
                    shared_edges.begin() + offset_edges, shared_edges.begin() + offset_edges + num_edges,
                    subgraph_edges.begin() + pos_m);
                std::copy(
                    shared_edge_weights.begin() + offset_edges, shared_edge_weights.begin() + offset_edges + num_edges,
                    subgraph_edge_weights.begin() + pos_m);

                // copied independent nodes arrays -- thus, offset segment by number of edges received from previous PEs
                for (NodeID u = 0; u < num_nodes; ++u) {
                    subgraph_nodes[pos_n + 1 + u] += pos_m;
                }

                pos_n += num_nodes;
                pos_m += num_edges;
            }
            offsets[b].push_back(pos_n);

            subgraphs[b] = {
                std::move(subgraph_nodes), std::move(subgraph_edges), std::move(subgraph_node_weights),
                std::move(subgraph_edge_weights), false};
        });
    }

    return {std::move(subgraphs), std::move(offsets)};
}
} // namespace

ExtractedSubgraphs distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph) {
    auto extracted_local_subgraphs     = extract_local_block_induced_subgraphs(p_graph);
    auto [gathered_subgraphs, offsets] = gather_block_induced_subgraphs(p_graph, extracted_local_subgraphs);

    return {std::move(gathered_subgraphs), std::move(offsets), std::move(extracted_local_subgraphs.mapping)};
}

DistributedPartitionedGraph copy_subgraph_partitions(
    DistributedPartitionedGraph p_graph, const std::vector<shm::PartitionedGraph>& p_subgraphs,
    const ExtractedSubgraphs& extracted_subgraphs) {
    const auto& offsets = extracted_subgraphs.subgraph_offsets;
    const auto& mapping = extracted_subgraphs.mapping;

    const PEID size = mpi::get_comm_size(p_graph.communicator());

    // Assume that all subgraph partitions have the same number of blocks
    KASSERT(!p_subgraphs.empty());
    const PEID k_multiplier = p_subgraphs.front().k();
    const PEID new_k        = p_graph.k() * k_multiplier;

    // Send new block IDs to the right PE
    std::vector<std::vector<BlockID>> partition_sendbufs(size);
    for (BlockID b = 0; b < p_subgraphs.size(); ++b) {
        const auto& p_subgraph = p_subgraphs[b];
        tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
            const NodeID from = offsets[b][pe];
            const NodeID to   = offsets[b][pe + 1];
            for (NodeID u = from; u < to; ++u) {
                partition_sendbufs[pe].push_back(p_subgraph.block(u));
            }
        });
    }

    const auto partition_recvbufs = mpi::sparse_alltoall_get<BlockID>(partition_sendbufs, p_graph.communicator());

    // To index partition_recvbufs, we need the number of nodes *on our PE* in each block
    // -> Compute this now
    // @todo could also be kept in memory when extracting the subgraphs
    parallel::vector_ets<NodeID> block_sizes_ets(p_graph.k());
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto& r) {
        auto& block_sizes = block_sizes_ets.local();
        for (NodeID u = r.begin(); u != r.end(); ++u) {
            ++block_sizes[p_graph.block(u)];
        }
    });
    const auto block_sizes = block_sizes_ets.combine(std::plus{});

    std::vector<NodeID> block_offsets(p_graph.k() + 1);
    parallel::prefix_sum(block_sizes.begin(), block_sizes.end(), block_offsets.begin() + 1);

    // Assign nodes in p_graph to new blocks
    const BlockID num_blocks_per_pe = p_graph.k() / size;

    auto compute_block_owner = [&](const BlockID b) {
        return static_cast<PEID>(b / num_blocks_per_pe);
    };

    auto partition = p_graph.take_partition(); // NOTE: do not use p_graph after this

    p_graph.pfor_nodes([&](const NodeID u) {
        const BlockID b                    = partition[u];
        const PEID    owner                = compute_block_owner(b);
        const BlockID first_block_on_owner = owner * num_blocks_per_pe;
        const BlockID block_offset         = block_offsets[b] - block_offsets[first_block_on_owner];
        const NodeID  mapped_u             = mapping[u]; // ID of u in its block-induced subgraph

        KASSERT(static_cast<BlockID>(owner) < partition_recvbufs.size());
        KASSERT(mapped_u - block_offset < partition_recvbufs[owner].size());
        const BlockID new_b = b * k_multiplier + partition_recvbufs[owner][mapped_u - block_offset];
        partition[u]        = new_b;
    });

    // Create partitioned graph with the new partition
    DistributedPartitionedGraph new_p_graph(&p_graph.graph(), new_k, std::move(partition));

    // Synchronize block assignment of ghost nodes
    synchronize_ghost_node_block_ids(new_p_graph);

    KASSERT(graph::debug::validate_partition(new_p_graph), "graph partition in inconsistent state", assert::heavy);
    return new_p_graph;
}
} // namespace dkaminpar::graph
