/*******************************************************************************
 * @file:   locking_clustering_contraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Contracts a clustering computed by \c LockingLabelPropagation.
 ******************************************************************************/
#include "dkaminpar/coarsening/sparse_clustering_contraction.h"

#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/coarsening/local_graph_contraction.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph.h"
#include "kaminpar/datastructure/rating_map.h"

namespace dkaminpar::coarsening {
namespace {
struct GlobalEdge {
  GlobalNodeID from;
  GlobalNodeID to;
  EdgeWeight weight;
};

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

auto migrate_edges(const DistributedGraph &graph, const GlobalClustering &clustering) {
  const PEID size = mpi::get_comm_size(graph.communicator());

  std::vector<tbb::concurrent_vector<GlobalEdge>> send_buffers(size);

  graph.pfor_nodes_range([&](const auto r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const auto u_global_label = clustering[u];
      if (graph.is_owned_global_node(u_global_label)) {
        continue;
      }
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

  return mpi::sparse_alltoall_get<GlobalEdge, scalable_vector>(real_send_buffers, graph.communicator());
}

std::tuple<NodeID, LocalClustering, GlobalClustering, std::unordered_map<GlobalNodeID, NodeID>>
build_local_clustering(const DistributedGraph &graph, const GlobalClustering &clustering) {
  SET_DEBUG(true);

  //  growt::GlobalNodeIDMap<GlobalNodeID> global_ghost_to_local{graph.ghost_n()};
  //  auto global_ghost_to_local_ets = growt::create_handle_ets(global_ghost_to_local);
  std::unordered_map<GlobalNodeID, NodeID> global_clustering_to_local_clustering;

  // find cluster IDs used by this PE
  for (const NodeID u : graph.all_nodes()) {
    global_clustering_to_local_clustering[clustering[u]] = 1;
  }

  // map local cluster IDs
  NodeID next_local_cluster = 0;
  NodeID next_ghost_cluster = graph.n();

  for (auto &it : global_clustering_to_local_clustering) {
    const auto cluster = it.first;
    if (graph.is_owned_global_node(cluster)) {
      it.second = next_local_cluster++;
    } else {
      it.second = next_ghost_cluster++;
    }
  }

  // remap ghost cluster IDs to next_local_cluster..n()
  for (auto it : global_clustering_to_local_clustering) {
    if (it.second >= graph.n()) {
      it.second -= graph.n() - next_local_cluster;
      ASSERT(it.second < graph.total_n());
    }
  }

  // build local clustering
  LocalClustering local_clustering(graph.total_n());
  GlobalClustering local_clustering_to_global_clustering(graph.total_n());
  for (const NodeID u : graph.all_nodes()) {
    ASSERT(u < local_clustering.size());
    ASSERT(u < clustering.size());
    local_clustering[u] = global_clustering_to_local_clustering[clustering[u]];
    local_clustering_to_global_clustering[local_clustering[u]] = clustering[u];
  }

  HEAVY_ASSERT([&] {
    for (const NodeID u : graph.all_nodes()) {
      ASSERT(local_clustering[u] < graph.total_n()) << V(local_clustering[u]) << V(u) << V(graph.n());
    }
  });

  return {next_local_cluster, std::move(local_clustering), std::move(local_clustering_to_global_clustering),
          std::move(global_clustering_to_local_clustering)};
}
} // namespace

contraction::SparseClusteringContractionResult contract_clustering_sparse(const DistributedGraph &graph,
                                                                          const GlobalClustering &clustering) {
  SET_DEBUG(true);
  ASSERT(graph.total_n() <= clustering.size());

  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // contract local part of the graph
  auto local_clustering_result = build_local_clustering(graph, clustering);
  auto local_cluster_count = std::get<0>(local_clustering_result);
  auto &local_clustering = std::get<1>(local_clustering_result);
  auto &local_clustering_to_global_clustering = std::get<2>(local_clustering_result);
  auto &global_clustering_to_local_clustering = std::get<3>(local_clustering_result);
  DBG << V(local_cluster_count) << V(local_clustering) << V(local_clustering_to_global_clustering);
  auto [local_c_graph, local_c_mapping, local_m_ctx] = contract_local_clustering(graph, local_clustering);

  local_c_graph.print();


  HEAVY_ASSERT([&] {
    for (const NodeID u : graph.nodes()) {
      ASSERT(local_c_mapping[u] == local_clustering[u]);
    }
  });

  // compute node distribution
  auto c_node_distribution =
      helper::create_distribution_from_local_count<GlobalNodeID>(local_cluster_count, graph.communicator());
  DBG << V(c_node_distribution);

  // send other PEs their edges
  struct Edge {
    GlobalNodeID from;
    EdgeWeight weight;
    GlobalNodeID to;
  };
  std::vector<scalable_vector<Edge>> out_msg(size);
  scalable_vector<helper::LocalToGlobalEdge> edge_list;

  for (const NodeID u : local_c_graph.nodes()) {
    const GlobalNodeID u_cluster = local_clustering_to_global_clustering[u];
    const bool owned = graph.is_owned_global_node(u_cluster);

    for (const auto [e, v] : graph.neighbors(u)) {
      const GlobalNodeID v_cluster = local_clustering_to_global_clustering[v];
      ASSERT(u_cluster != v_cluster);

      const EdgeWeight e_weight = graph.edge_weight(e);
      if (owned) {
        edge_list.emplace_back(global_clustering_to_local_clustering[u_cluster], e_weight, v_cluster);
      }

      if (!graph.is_owned_global_node(v_cluster)) {
        const PEID owner = graph.find_owner_of_global_node(v_cluster);
        out_msg[owner].emplace_back(u_cluster, graph.edge_weight(e), v_cluster);
      }
    }
  }

  // exchange messages
  mpi::sparse_alltoall<Edge, scalable_vector>(
      out_msg,
      [&](const auto buffer, const PEID pe) {
        for (const auto &[from, weight, to] : buffer) {
          edge_list.emplace_back(global_clustering_to_local_clustering[from], weight, to);
        }
      },
      graph.communicator(), false);

  // update node weights
  /*
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
   */

  auto c_graph = coarsening::helper::build_distributed_graph_from_edge_list(
      edge_list, std::move(c_node_distribution), graph.communicator(), [&](const NodeID u) { return 0; });

  return {std::move(c_graph), {}};
}
} // namespace dkaminpar::coarsening
