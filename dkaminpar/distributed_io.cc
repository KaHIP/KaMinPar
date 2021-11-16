/*******************************************************************************
 * @file:   distributed_io.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Load distributed grpah from a single METIS file, node or edge
 * balanced.
 ******************************************************************************/
#include "dkaminpar/distributed_io.h"

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"
#include "dkaminpar/utility/math.h"
#include "kaminpar/io.h"
#include "kaminpar/utility/strings.h"

#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>

namespace dkaminpar::io {
SET_DEBUG(false);

DistributedGraph read_node_balanced(const std::string &filename) {
  if (shm::utility::str::ends_with(filename, "bgf") || shm::utility::str::ends_with(filename, "bin")) {
    DBG << "Expect binary graph format";
    return binary::read_node_balanced(filename);
  }

  DBG << "Expect Metis graph format";
  return metis::read_node_balanced(filename);
}

namespace metis {
namespace shm = kaminpar::io::metis;

DistributedGraph read_node_balanced(const std::string &filename) {
  const auto comm_info = mpi::get_comm_info();
  const PEID size = comm_info.first;
  const PEID rank = comm_info.second;

  graph::Builder builder{};

  GlobalNodeID current = 0;
  GlobalNodeID from = 0;
  GlobalNodeID to = 0;

  shm::read_observable(
      filename,
      [&](const auto &format) {
        const auto global_n = static_cast<GlobalNodeID>(format.number_of_nodes);
        const auto global_m = static_cast<GlobalNodeID>(format.number_of_edges) * 2;
        DBG << "Loading graph with global_n=" << global_n << " and global_m=" << global_m;

        scalable_vector<GlobalNodeID> node_distribution(size + 1);
        for (PEID p = 0; p < size; ++p) {
          const auto [p_from, p_to] = math::compute_local_range<GlobalNodeID>(global_n, size, p);
          node_distribution[p + 1] = p_to;
        }
        ASSERT(node_distribution.front() == 0);
        ASSERT(node_distribution.back() == global_n);

        from = node_distribution[rank];
        to = node_distribution[rank + 1];
        DBG << "PE " << rank << ": from=" << from << " to=" << to << " n=" << format.number_of_nodes;

        builder.initialize(global_n, global_m, rank, std::move(node_distribution));
      },
      [&](const std::uint64_t &u_weight) {
        ++current;
        if (current > to) {
          return false;
        }
        if (current > from) {
          builder.create_node(static_cast<NodeWeight>(u_weight));
        }
        return true;
      },
      [&](const std::uint64_t &e_weight, const std::uint64_t &v) {
        if (current > from) {
          builder.create_edge(static_cast<EdgeWeight>(e_weight), static_cast<GlobalNodeID>(v));
        }
      });

  return builder.finalize();
}

DistributedGraph read_edge_balanced(const std::string &filename) {
  const auto comm_info = mpi::get_comm_info();
  const PEID size = comm_info.first;
  const PEID rank = comm_info.second;

  PEID current_pe = 0;
  GlobalNodeID current_node = 0;
  GlobalEdgeID current_edge = 0;
  GlobalEdgeID to = 0;

  scalable_vector<EdgeID> nodes;
  scalable_vector<GlobalNodeID> global_edges;
  scalable_vector<NodeWeight> node_weights;
  scalable_vector<EdgeWeight> edge_weights;
  scalable_vector<PEID> ghost_owner;
  scalable_vector<GlobalNodeID> ghost_to_global;
  std::unordered_map<GlobalNodeID, NodeID> global_to_ghost;

  scalable_vector<GlobalNodeID> node_distribution(size + 1);
  scalable_vector<GlobalEdgeID> edge_distribution(size + 1);

  // read graph file
  shm::read_observable(
      filename,
      [&](const auto &format) {
        const auto global_n = static_cast<GlobalNodeID>(format.number_of_nodes);
        const auto global_m = static_cast<GlobalEdgeID>(format.number_of_edges) * 2;
        node_distribution.back() = global_n;
        edge_distribution.back() = global_m;
        const auto [pe_from, pe_to] = math::compute_local_range<GlobalEdgeID>(global_m, size, current_pe);
        to = pe_to;
      },
      [&](const std::uint64_t &u_weight) {
        if (current_edge >= to) {
          node_distribution[current_pe] = current_node;
          edge_distribution[current_pe] = current_edge;
          ++current_pe;

          const GlobalEdgeID global_m = edge_distribution.back();
          const auto [pe_from, pe_to] = math::compute_local_range<GlobalEdgeID>(global_m, size, current_pe);
          to = pe_to;
        }

        if (current_pe == rank) {
          nodes.push_back(global_edges.size());
          node_weights.push_back(static_cast<NodeWeight>(u_weight));
        }

        ++current_node;
        return true;
      },
      [&](const std::uint64_t &e_weight, const std::uint64_t &v) {
        if (current_pe == rank) {
          global_edges.push_back(static_cast<GlobalNodeID>(v));
          edge_weights.push_back(static_cast<EdgeWeight>(e_weight));
        }
        ++current_edge;
      });

  // at this point we should have a valid node and edge distribution
  const GlobalNodeID offset_n = node_distribution[rank];
  const auto local_n = static_cast<NodeID>(node_distribution[rank + 1] - node_distribution[rank]);

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

        auto it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), global_v);
        const auto owner = static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
        ghost_owner.push_back(owner);
      }

      edges[i] = global_to_ghost[global_v];
    }
  }

  // init graph
  return {std::move(node_distribution),
          std::move(edge_distribution),
          std::move(nodes),
          std::move(edges),
          std::move(node_weights),
          std::move(edge_weights),
          std::move(ghost_owner),
          std::move(ghost_to_global),
          std::move(global_to_ghost),
          MPI_COMM_WORLD};
}

void write(const std::string &filename, const DistributedGraph &graph, const bool write_node_weights,
           const bool write_edge_weights) {
  { std::ofstream tmp(filename); } // clear file

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

    for (const NodeID u : graph.nodes()) {
      if (write_node_weights) {
        out << graph.node_weight(u) << " ";
      }
      for (const auto [e, v] : graph.neighbors(u)) {
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

DistributedGraph read_node_balanced(const std::string &filename) {
  std::ifstream in(filename);

  // read header
  IDType version, global_n, global_m;
  in.read(reinterpret_cast<char *>(&version), sizeof(IDType));
  ALWAYS_ASSERT(version == 3) << "invalid binary graph format!";

  in.read(reinterpret_cast<char *>(&global_n), sizeof(IDType));
  in.read(reinterpret_cast<char *>(&global_m), sizeof(IDType));

  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  const auto [from, to] = math::compute_local_range<GlobalNodeID>(global_n, size, rank);
  const NodeID n = static_cast<NodeID>(to - from);

  // read nodes
  scalable_vector<EdgeID> nodes(n + 1);
  IDType first_edge_index = 0;
  IDType first_invalid_edge_index = 0;
  {
    // read part of global nodes array
    scalable_vector<IDType> global_nodes(n + 1);
    const std::streamsize offset = 3 * sizeof(IDType) + from * sizeof(IDType);
    const std::streamsize length = (n + 1) * sizeof(IDType);
    in.seekg(offset);
    in.read(reinterpret_cast<char *>(global_nodes.data()), length);

    // build local nodes array
    first_edge_index = global_nodes.front();
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
  const std::streamsize offset = first_edge_index;
  const std::streamsize length = first_invalid_edge_index - first_edge_index;
  in.seekg(offset);
  in.read(reinterpret_cast<char *>(global_edges.data()), length);

  // translate to local edges
  Atomic<NodeID> next_ghost_node_id = n;
  using GhostNodeFilter = tbb::concurrent_hash_map<GlobalNodeID, NodeID>;
  GhostNodeFilter discovered_ghost_nodes_filter;

  tbb::parallel_for<std::size_t>(0, global_edges.size(),
                                 [from = from, to = to, &global_edges, &edges, &next_ghost_node_id,
                                  &discovered_ghost_nodes_filter](const std::size_t i) {
                                   const GlobalNodeID edge_target = global_edges[i];
                                   if (from <= edge_target && edge_target < to) { // local node
                                     edges[i] = static_cast<NodeID>(edge_target - from);
                                   } else { // ghost node
                                     GhostNodeFilter::accessor accessor;
                                     if (discovered_ghost_nodes_filter.insert(accessor, edge_target)) {
                                       const NodeID ghost_node_id =
                                           next_ghost_node_id.fetch_add(1, std::memory_order_relaxed);
                                       accessor->second = ghost_node_id;
                                     }
                                   }
                                 });

  auto node_distribution = mpi::build_distribution_from_local_count<GlobalNodeID, scalable_vector>(n, MPI_COMM_WORLD);
  auto edge_distribution = mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(m, MPI_COMM_WORLD);

  // remap ghost nodes
  const NodeID ghost_n = static_cast<NodeID>(discovered_ghost_nodes_filter.size());
  scalable_vector<GlobalNodeID> ghost_to_global(ghost_n);
  growt::StaticGhostNodeMapping global_to_ghost(ghost_n);
  scalable_vector<PEID> ghost_owner(discovered_ghost_nodes_filter.size());

  tbb::parallel_for(discovered_ghost_nodes_filter.range(), [&](const auto r) {
    for (auto it = r.begin(); it != r.end(); ++it) {
      const GlobalNodeID global_node_id = it->first;
      const NodeID local_node_id = it->second;
      const NodeID local_ghost_id = local_node_id - n;
      ASSERT(local_ghost_id < discovered_ghost_nodes_filter.size());

      ghost_to_global[local_ghost_id] = global_node_id;
      global_to_ghost.insert(global_node_id + 1, local_node_id); // 0 cannot be used as key

      // find ghost node owner using binary search
      const auto owner_it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), global_node_id);
      ghost_owner[local_ghost_id] = static_cast<PEID>(std::distance(node_distribution.begin(), owner_it) - 1);
    }
  });

  // copy edges to ghost nodes to local edges array
  tbb::parallel_for<std::size_t>(0, global_edges.size(),
                                 [from = from, to = to, &edges, &global_edges, &global_to_ghost](const std::size_t i) {
                                   const GlobalNodeID edge_target = global_edges[i];
                                   if (edge_target < from || edge_target >= to) {
                                     edges[i] = static_cast<NodeID>((*global_to_ghost.find(edge_target + 1)).second);
                                   }
                                 });

  // binary input graph is always unweighted -- allocate dummy vectors
  scalable_vector<NodeWeight> node_weights(next_ghost_node_id, 1);
  scalable_vector<EdgeWeight> edge_weights(m, 1);

  return {std::move(node_distribution),
          std::move(edge_distribution),
          std::move(nodes),
          std::move(edges),
          std::move(node_weights),
          std::move(edge_weights),
          std::move(ghost_owner),
          std::move(ghost_to_global),
          std::move(global_to_ghost),
          MPI_COMM_WORLD};
}
} // namespace binary
} // namespace dkaminpar::io