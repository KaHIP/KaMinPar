/*******************************************************************************
 * @file:   locking_clustering_contraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Contracts a clustering computed by \c LockingLabelPropagation.
 ******************************************************************************/
#include "dkaminpar/algorithm/locking_clustering_contraction.h"

#include "dkaminpar/algorithm/local_graph_contraction.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph.h"
#include "kaminpar/datastructure/rating_map.h"

namespace dkaminpar::graph {
namespace {
using LocalClusterArray = scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>>;
using GlobalClusterArray = LockingLpClustering::AtomicClusterArray;

struct GlobalEdge {
  GlobalNodeID from;
  GlobalNodeID to;
  EdgeWeight weight;
};

#ifdef KAMINPAR_ENABLE_ASSERTIONS
bool CHECK_CLUSTERING_INVARIANT(const DistributedGraph &graph,
                                const LockingLpClustering::AtomicClusterArray &clustering) {
  mpi::graph::sparse_alltoall_custom<GlobalNodeID>(
      graph, 0, graph.n(),
      [&](const NodeID u) {
        ASSERT(clustering[u] < graph.global_n());
        return !graph.is_owned_global_node(clustering[u]);
      },
      [&](const NodeID u) { return std::make_pair(clustering[u], graph.find_owner_of_global_node(clustering[u])); },
      [&](const auto &buffer, const PEID pe) {
        for (const GlobalNodeID label : buffer) {
          ASSERT(graph.is_owned_global_node(label));
          const NodeID local_label = graph.global_to_local_node(label);
          ASSERT(clustering[local_label] == label)
              << "from PE: " << pe << " has nodes in cluster " << label << ", but local node " << local_label
              << " is in cluster " << clustering[local_label];
        }
      });

  return true;
}
#endif // KAMINPAR_ENABLE_ASSERTIONS

std::pair<GlobalNodeID, scalable_vector<GlobalNodeID>> compute_coarse_node_distribution(const DistributedGraph &graph,
                                                                                        const NodeID c_n) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // Compute new node distribution, total number of coarse nodes
  const GlobalNodeID last_node = mpi::scan(static_cast<GlobalNodeID>(c_n), MPI_SUM, graph.communicator());
  [[maybe_unused]] const GlobalNodeID first_node = last_node - c_n;
  scalable_vector<GlobalNodeID> c_node_distribution(size + 1);
  c_node_distribution[rank + 1] = last_node;
  mpi::allgather(&c_node_distribution[rank + 1], 1, c_node_distribution.data() + 1, 1, graph.communicator());
  const GlobalNodeID c_global_n = c_node_distribution.back();

  return {c_global_n, std::move(c_node_distribution)};
}

auto migrate_edges(const DistributedGraph &graph, const LockingLpClustering::AtomicClusterArray &clustering) {
  const PEID size = mpi::get_comm_size(graph.communicator());

  std::vector<tbb::concurrent_vector<GlobalEdge>> send_buffers(size);

  graph.pfor_nodes_range([&](const auto r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const auto u_global_label = clustering[u];
      if (graph.is_owned_global_node(u_global_label)) { continue; }
      const PEID u_pe = graph.find_owner_of_global_node(u_global_label);

      // node migrates to another PE
      for (const auto [e, v] : graph.neighbors(u)) {
        const auto v_global_label = clustering[v];
        if (u_global_label != v_global_label) {
          send_buffers[u_pe].push_back({.from = u_global_label, .to = v_global_label, .weight = graph.edge_weight(e)});
        }
      }
    }
  });

  std::vector<scalable_vector<GlobalEdge>> real_send_buffers(size);
  for (PEID pe = 0; pe < size; ++pe) {
    std::copy(send_buffers[pe].begin(), send_buffers[pe].end(), real_send_buffers[pe].begin());
  }

  return mpi::sparse_all_to_all_get<scalable_vector>(real_send_buffers, 0, graph.communicator());
}

std::pair<LocalClusterArray, std::unordered_map<GlobalNodeID, NodeID>>
build_local_clustering(const DistributedGraph &graph, const GlobalClusterArray &clustering) {
  SET_DEBUG(true);

  //  growt::GlobalNodeIDMap<GlobalNodeID> global_ghost_to_local{graph.ghost_n()};
  //  auto global_ghost_to_local_ets = growt::create_handle_ets(global_ghost_to_local);
  std::unordered_map<GlobalNodeID, NodeID> global_to_local;

  for (const NodeID u : graph.nodes()) { global_to_local[clustering[u]] = 1; }

  NodeID next_id = 0;
  { // compute prefix sum
    for (const auto it : global_to_local) { global_to_local[it.first] = next_id++; }
  }

  LocalClusterArray local_clustering(graph.total_n());
  for (const NodeID u : graph.nodes()) {
    ASSERT(u < local_clustering.size());
    ASSERT(u < clustering.size());
    local_clustering[u] = global_to_local[clustering[u]];
  }

  HEAVY_ASSERT([&] {
    for (const NodeID u : graph.nodes()) {
      ASSERT(local_clustering[u] < graph.n()) << V(local_clustering[u]) << V(u) << V(graph.n());
    }
  });

  return {std::move(local_clustering), std::move(global_to_local)};
}
} // namespace

contraction::LockingClusteringContractionResult
contract_locking_clustering(const DistributedGraph &graph, const LockingLpClustering::AtomicClusterArray &clustering,
                            contraction::MemoryContext m_ctx) {
  SET_DEBUG(true);
  HEAVY_ASSERT(CHECK_CLUSTERING_INVARIANT(graph, clustering));
  ASSERT(graph.total_n() <= clustering.size());

  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // contract local part of the graph
  auto [local_clustering, global_to_local_clustering] = build_local_clustering(graph, clustering);
  SLOG << V(local_clustering);
  auto [local_c_graph, local_c_mapping, local_m_ctx] = contract_local_clustering(graph, local_clustering,
                                                                                 std::move(m_ctx));
  m_ctx = std::move(local_m_ctx);

  // build map from coarse node to global cluster
  scalable_vector<GlobalNodeID> c_node_to_global(local_c_graph.total_n(), kInvalidGlobalNodeID);
  for (const NodeID u : graph.all_nodes()) {
    const NodeID c_u = local_c_mapping[u];
    ASSERT(c_node_to_global[c_u] == kInvalidGlobalNodeID || c_node_to_global[c_u] == clustering[u]);
    c_node_to_global[c_u] = clustering[u];
  }

  HEAVY_ASSERT([&] {
    for (const NodeID c_u : local_c_graph.all_nodes()) { ASSERT(c_node_to_global[c_u] != kInvalidGlobalNodeID); }
  });

  LOG << V(c_node_to_global);

  // send coarse edges that belong to other PEs
  // look for coarse nodes that we want to drop
  scalable_vector<bool> drop_node(local_c_graph.total_n(), true);
  std::vector<scalable_vector<GlobalEdge>> migrate_edges_send_buffers(size);

  for (const NodeID c_u : local_c_graph.nodes()) {
    const auto c_u_global = c_node_to_global[c_u];
    const bool c_u_is_local_node = graph.is_owned_global_node(c_u_global);

    if (c_u_is_local_node) {
      drop_node[c_u] = false;

      for (const NodeID c_v : local_c_graph.adjacent_nodes(c_u)) {
        const bool c_v_is_local_node = graph.is_owned_global_node(c_node_to_global[c_v]);
        if (!c_v_is_local_node) { drop_node[c_v] = false; } // keep as ghost node
      }
    } else {
      const PEID c_u_owner = graph.find_owner_of_global_node(c_u_global);

      for (const auto [e, c_v] : local_c_graph.neighbors(c_u)) {
        const auto c_v_global = c_node_to_global[c_v];

        if (c_v_global != c_u_global) {
          migrate_edges_send_buffers[c_u_owner].push_back({
              .from = c_u_global,
              .to = c_v_global,
              .weight = local_c_graph.edge_weight(e),
          });
        }
      }
    }
  }
  auto migrate_edges_recv_buffers = mpi::sparse_all_to_all_get<scalable_vector>(migrate_edges_send_buffers, 0,
                                                                                graph.communicator());

  // update node weights
  struct UpdateNodeWeight {
    GlobalNodeID label;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_custom<UpdateNodeWeight>(
      graph, 0, graph.n(), [&](const NodeID u) { return !graph.is_owned_global_node(clustering[u]); },
      [&](const NodeID u) -> std::pair<UpdateNodeWeight, PEID> {
        const auto u_label = graph.local_to_global_node(u);
        const PEID pe = graph.find_owner_of_global_node(u_label);
        return {{.label = u_label, .weight = graph.node_weight(u)}, pe};
      },
      [&](const auto) {
//        for (const auto [label, weight] : buffer) {
//           increase label by weight
//        }
      });

  // build coarse graph

  DBG << V(drop_node);

  shm::parallel::parallel_for_over_chunks(migrate_edges_recv_buffers, [&](const GlobalEdge &edge) {
    DBG << V(edge.from) << V(edge.to) << V(edge.weight);
  });

  return {std::move(local_c_graph), std::move(local_c_mapping), std::move(m_ctx)};
}
} // namespace dkaminpar::graph
