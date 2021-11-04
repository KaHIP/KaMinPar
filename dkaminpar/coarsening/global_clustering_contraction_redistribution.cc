/*******************************************************************************
 * @file:   global_clustering_contraction_redistribution.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Shared-memory parallel contraction of global clustering without
 * any restrictions.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_clustering_contraction_redistribution.h"

#include "dkaminpar/coarsening/coarsening.h"
#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/mpi_wrapper.h"
#include "dkaminpar/utility/math.h"
#include "dkaminpar/utility/vector_ets.h"

#include <tbb/concurrent_hash_map.h>

namespace dkaminpar::coarsening {
SET_DEBUG(false);

namespace {
// global mapping, global number of coarse nodes
std::pair<scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>>, GlobalNodeID>
compute_mapping(const DistributedGraph &graph,
                const scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>> &clustering) {
  const auto size = mpi::get_comm_size(graph.communicator());
  const auto rank = mpi::get_comm_rank(graph.communicator());

  // mark global node IDs that are used as cluster IDs
  std::vector<tbb::concurrent_hash_map<NodeID, NodeID>> used_clusters_map(size);
  std::vector<shm::parallel::IntegralAtomicWrapper<NodeID>> next_slot_for_pe(size);

  graph.pfor_nodes_range([&](const auto r) {
    tbb::concurrent_hash_map<NodeID, NodeID>::accessor accessor;

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const GlobalNodeID u_cluster = clustering[u];

      PEID u_cluster_owner;
      NodeID u_local_cluster;
      if (graph.is_owned_global_node(u_cluster)) {
        u_cluster_owner = rank;
        u_local_cluster = graph.global_to_local_node(u_cluster);
      } else {
        u_cluster_owner = graph.find_owner_of_global_node(u_cluster);
        u_local_cluster = static_cast<NodeID>(u_cluster - graph.offset_n(u_cluster_owner));
      }

      if (used_clusters_map[u_cluster_owner].insert(accessor, u_local_cluster)) {
        accessor->second = next_slot_for_pe[u_cluster_owner]++;
      }
    }
  });

  // used_clusters_vec[pe] holds local node IDs of PE pe that are used as cluster IDs on this PE
  std::vector<scalable_vector<NodeID>> used_clusters_vec(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    used_clusters_vec[pe].resize(used_clusters_map[pe].size());
    tbb::parallel_for(used_clusters_map[pe].range(), [&](const auto r) {
      for (auto it = r.begin(); it != r.end(); ++it) {
        used_clusters_vec[pe][it->second] = it->first;
      }
    });
  });

  // send each PE its local node IDs that are used as cluster IDs somewhere
  const auto in_msg = mpi::sparse_alltoall_get<NodeID, scalable_vector>(used_clusters_vec, graph.communicator(), true);

  // map local labels to consecutive coarse node IDs
  scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>> label_mapping(graph.total_n());
  shm::parallel::parallel_for_over_chunks(in_msg, [&](const NodeID local_label) {
    ASSERT(local_label < graph.n());
    label_mapping[local_label].store(1, std::memory_order_relaxed);
  });
  shm::parallel::prefix_sum(label_mapping.begin(), label_mapping.end(), label_mapping.begin());

  const NodeID c_label_n = label_mapping.empty() ? 0 : static_cast<NodeID>(label_mapping.back());
  const GlobalNodeID c_label_from = mpi::exscan<GlobalNodeID>(c_label_n, MPI_SUM, graph.communicator());
  const auto c_label_distribution = mpi::allgather(c_label_from, graph.communicator());
  const GlobalNodeID c_global_n = mpi::allreduce(c_label_n, MPI_SUM, graph.communicator());

  // send mapping to other PEs that use cluster IDs from this PE -- i.e., answer in_msg
  std::vector<scalable_vector<NodeID>> out_msg(size);
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    out_msg[pe].resize(in_msg[pe].size());
    tbb::parallel_for<std::size_t>(0, in_msg[pe].size(), [&](const std::size_t i) {
      ASSERT(in_msg[pe][i] < label_mapping.size());
      out_msg[pe][i] = label_mapping[in_msg[pe][i]] - 1; // label_mapping is 1-based due to the prefix sum operation
    });
  });

  const auto label_remap = mpi::sparse_alltoall_get<NodeID, scalable_vector>(out_msg, graph.communicator(), true);

  // now  we use label_mapping as a [fine node -> coarse node] mapping of local nodes on this PE -- and extend it
  // for ghost nodes in the next step
  // all cluster[.] labels are stored in label_remap, thus we can overwrite label_mapping
  graph.pfor_nodes([&](const NodeID u) {
    const GlobalNodeID u_cluster = clustering[u];
    PEID u_cluster_owner;
    NodeID u_local_cluster;

    if (graph.is_owned_global_node(u_cluster)) {
      u_cluster_owner = rank;
      u_local_cluster = graph.global_to_local_node(u_cluster);
    } else {
      u_cluster_owner = graph.find_owner_of_global_node(u_cluster);
      u_local_cluster = static_cast<NodeID>(u_cluster - graph.offset_n(u_cluster_owner));
    }

    tbb::concurrent_hash_map<NodeID, NodeID>::accessor accessor;
    [[maybe_unused]] const bool found = used_clusters_map[u_cluster_owner].find(accessor, u_local_cluster);
    ASSERT(found) << V(u_local_cluster) << V(u_cluster_owner) << V(u) << V(u_cluster);

    const NodeID slot_in_msg = accessor->second;
    label_mapping[u] = c_label_distribution[u_cluster_owner] + label_remap[u_cluster_owner][slot_in_msg];
  });

  // ASSERT label_mapping[0..n] maps to coarse node IDs
  return {std::move(label_mapping), c_global_n};
}

void exchange_ghost_node_mapping(const DistributedGraph &graph, auto &label_mapping) {
  struct Message {
    NodeID local_node;
    GlobalNodeID coarse_global_node;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message, std::vector>(
      graph,
      [&](const NodeID u) -> Message {
        return {u, label_mapping[u]};
      },
      [&](const auto buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          const auto &[local_node_on_other_pe, coarse_global_node] = buffer[i];
          const auto local_node = graph.global_to_local_node(graph.offset_n(pe) + local_node_on_other_pe);
          label_mapping[local_node] = coarse_global_node;
        });
      });
}

DistributedGraph build_coarse_graph(const DistributedGraph &graph, const auto &mapping, const GlobalNodeID c_global_n) {
  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  // compute coarse node distribution
  auto c_node_distribution =
      helper::create_perfect_distribution_from_global_count<GlobalNodeID>(c_global_n, graph.communicator());
  const auto from = c_node_distribution[rank];
  const auto to = c_node_distribution[rank + 1];

  // lambda to map global coarse node IDs to their owner PE
  auto compute_coarse_node_owner = [size = size, c_global_n](const GlobalNodeID coarse_global_node) -> PEID {
    ASSERT(coarse_global_node < c_global_n);
    return math::compute_local_range_rank<PEID>(c_global_n, size, coarse_global_node);
  };

  // next, send each PE the edges it owns in the coarse graph
  // first, count the number of edges for each PE
  parallel::vector_ets<EdgeID> num_edges_for_pe_ets(size);
  graph.pfor_nodes_range([&](const auto r) {
    auto &num_edges_for_pe = num_edges_for_pe_ets.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      ASSERT(u < mapping.size());
      const auto c_u = mapping[u];
      const auto c_u_owner = compute_coarse_node_owner(c_u);
      ASSERT(static_cast<std::size_t>(c_u_owner) < num_edges_for_pe.size());

      for (const auto [e, v] : graph.neighbors(u)) {
        if (c_u != mapping[v]) { // ignore self loops
          num_edges_for_pe[c_u_owner]++;
        }
      }
    }
  });
  auto num_edges_for_pe = num_edges_for_pe_ets.combine(std::plus{});

  // allocate memory for edge messages
  struct EdgeMessage {
    NodeID u;
    EdgeWeight weight;
    GlobalNodeID v;
  };

  std::vector<scalable_vector<EdgeMessage>> out_msg;
  for (PEID pe = 0; pe < size; ++pe) {
    out_msg.emplace_back(num_edges_for_pe[pe]);
  }

  // create messages
  std::vector<shm::parallel::IntegralAtomicWrapper<EdgeID>> next_out_msg_slot(size);
  graph.pfor_nodes([&](const NodeID u) {
    const auto c_u = mapping[u];
    const auto c_u_owner = compute_coarse_node_owner(c_u);
    const auto local_c_u = static_cast<NodeID>(c_u - c_node_distribution[c_u_owner]);

    for (const auto [e, v] : graph.neighbors(u)) {
      const auto c_v = mapping[v];

      if (c_u != c_v) { // ignore self loops
        const std::size_t slot = next_out_msg_slot[c_u_owner].fetch_add(1, std::memory_order_relaxed);
        out_msg[c_u_owner][slot] = {.u = local_c_u, .weight = graph.edge_weight(e), .v = c_v};
        DBG << "--> " << c_u_owner << ": [" << slot << "]={.u=" << local_c_u << ", .weight=" << graph.edge_weight(e)
            << ", .v=" << c_v << "}";
      }
    }
  });

  ASSERT([&] {
    for (PEID pe = 0; pe < size; ++pe) {
      ASSERT(next_out_msg_slot[pe] == num_edges_for_pe[pe]);
    }
  });

  // exchange messages
  const auto in_msg = mpi::sparse_alltoall_get<EdgeMessage, scalable_vector>(out_msg, graph.communicator(), true);

  // Copy edge lists to a single list and free old list
  EdgeID total_num_edges = 0;
  for (const auto &list : in_msg) {
    total_num_edges += list.size();
  }
  scalable_vector<EdgeMessage> edge_list(total_num_edges);
  {
    EdgeID pos = 0;
    for (const auto &list : in_msg) {
      std::copy(list.begin(), list.end(), edge_list.begin() + pos);
      pos += list.size();
    }

    // free in_msg
    std::vector<scalable_vector<EdgeMessage>> tmp = std::move(in_msg);
  }

  // TODO since we do not know the number of coarse ghost nodes yet, allocate memory only for local nodes and
  // TODO resize in build_distributed_graph_from_edge_list
  ASSERT(from <= to);
  scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeWeight>> node_weights(to - from);
  struct NodeWeightMessage {
    NodeID node;
    NodeWeight weight;
  };

  // TODO accumulate node weights before sending them -> no longer need an atomic
  mpi::graph::sparse_alltoall_custom<NodeWeightMessage>(
      graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER,
      [&](const NodeID u) -> std::pair<NodeWeightMessage, PEID> {
        const auto c_u = mapping[u];
        const PEID c_u_owner = compute_coarse_node_owner(c_u);
        const NodeID c_u_local = c_u - c_node_distribution[c_u_owner];
        return {{c_u_local, graph.node_weight(u)}, c_u_owner};
      },
      [&](const auto r) {
        tbb::parallel_for<std::size_t>(0, r.size(), [&](const std::size_t i) {
          node_weights[r[i].node].fetch_add(r[i].weight, std::memory_order_relaxed);
        });
      },
      true);

  // now every PE has an edge list with all edges -- so we can build the graph from it
  return helper::build_distributed_graph_from_edge_list(edge_list, std::move(c_node_distribution), graph.communicator(),
                                                        [&](const NodeID u) {
                                                          ASSERT(u < node_weights.size());
                                                          return node_weights[u].load(std::memory_order_relaxed);
                                                        });
}

void update_ghost_node_weights(DistributedGraph &graph) {
  struct Message {
    GlobalNodeID global_node;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message, std::vector>(
      graph,
      [&](const NodeID u, const PEID pe) -> Message {
        DBG << V(pe) << V(u) << V(graph.local_to_global_node(u)) << V(graph.node_weight(u))
            << V(graph.contains_local_node(u));
        return {graph.local_to_global_node(u), graph.node_weight(u)};
      },
      [&](const auto buffer) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          const auto &message = buffer[i];
          const NodeID local_node = graph.global_to_local_node(message.global_node);
          graph.set_ghost_node_weight(local_node, message.weight);
        });
      });
}
} // namespace

RedistributedGlobalContractionResult contract_global_clustering_redistribute(const DistributedGraph &graph,
                                                                             const GlobalClustering &clustering) {
  // compute local mapping for owned nodes
  auto [mapping, c_global_n] = compute_mapping(graph, clustering);

  // compute local mapping for ghost nodes
  exchange_ghost_node_mapping(graph, mapping);

  // build coarse graph
  auto c_graph = build_coarse_graph(graph, mapping, c_global_n);
  update_ghost_node_weights(c_graph);

  return {std::move(c_graph), std::move(mapping)};
}
} // namespace dkaminpar::coarsening