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
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/mpi_wrapper.h"
#include "dkaminpar/utility/math.h"

#include <tbb/concurrent_hash_map.h>

namespace dkaminpar::coarsening {
SET_DEBUG(true);

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

  const NodeID c_label_n = static_cast<NodeID>(label_mapping.back());
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

    ASSERT(u_cluster_owner < size);
    ASSERT(u_cluster_owner < used_clusters_map.size());

    tbb::concurrent_hash_map<NodeID, NodeID>::accessor accessor;
    [[maybe_unused]] const bool found = used_clusters_map[u_cluster_owner].find(accessor, u_local_cluster);
    ASSERT(found) << V(u_local_cluster) << V(u_cluster_owner) << V(u) << V(u_cluster);

    const NodeID slot_in_msg = accessor->second;

    ASSERT(u < label_mapping.size());
    ASSERT(u_cluster_owner < label_remap.size());
    ASSERT(slot_in_msg < label_remap[u_cluster_owner].size());
    label_mapping[u] = c_label_distribution[u_cluster_owner] + label_remap[u_cluster_owner][slot_in_msg];
  });

  // ASSERT label_mapping[0..n] maps to coarse node IDs
  return {std::move(label_mapping), c_global_n};
}

void exchange_ghost_node_mapping(const DistributedGraph &graph, auto &label_mapping, auto &clustering) {
  struct Message {
    GlobalNodeID global_node;
    GlobalNodeID coarse_global_node;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message, std::vector>(
      graph,
      [&](const NodeID u) -> Message {
        return {graph.local_to_global_node(u), label_mapping[u]};
      },
      [&](const auto buffer) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          const auto &message = buffer[i];
          const auto local_node = graph.global_to_local_node(message.global_node);
          label_mapping[local_node] = message.coarse_global_node;
        });
      });
}

struct HashedEdge {
  GlobalNodeID u;
  GlobalNodeID v;
};

struct HashedEdgeComparator {
  bool operator()(const HashedEdge &e1, const HashedEdge &e2) const {
    return (e1.u == e2.u && e1.v == e2.v) || (e1.u == e2.v && e1.v == e2.u);
  }
};

struct HashedEdgeHash {
  GlobalNodeID operator()(const HashedEdge &e) const { return e.u ^ e.v; }
};

struct HashedGraph {
  using EdgeMap = std::unordered_map<HashedEdge, GlobalEdgeWeight, HashedEdgeHash, HashedEdgeComparator>;
  using NodeMap = std::unordered_map<GlobalNodeID, GlobalNodeWeight>;

  NodeMap nodes;
  EdgeMap edges;
};

HashedGraph hash_local_graph(const DistributedGraph &graph, auto &label_mapping) {
  HashedGraph h_graph;

  for (const NodeID u : graph.nodes()) {
    const GlobalNodeID c_u = label_mapping[u];
    h_graph.nodes[c_u] += graph.node_weight(u);

    for (const auto [e, v] : graph.neighbors(u)) {
      const GlobalNodeID c_v = label_mapping[v];
      if (c_v != c_u) {
        DBG << "Edge " << c_u << " <-> " << c_v;
        h_graph.edges[{c_u, c_v}] += graph.edge_weight(e);
      }
    }
  }

  return h_graph;
}

scalable_vector<GlobalNodeID> compute_coarse_node_distribution(const NodeID c_global_n, MPI_Comm comm) {
  const auto size = mpi::get_comm_size(comm);

  scalable_vector<GlobalNodeID> c_node_distribution(size + 1);
  for (PEID pe = 0; pe < size; ++pe) {
    c_node_distribution[pe + 1] = math::compute_local_range<GlobalNodeID>(c_global_n, size, pe).second;
  }

  return c_node_distribution;
}

DistributedGraph build_coarse_graph(const DistributedGraph &graph, HashedGraph &h_graph,
                                    const GlobalNodeID c_global_n) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // compute coarse node distribution
  auto c_node_distribution = compute_coarse_node_distribution(c_global_n, graph.communicator());
  const auto from = c_node_distribution[rank];
  const auto to = c_node_distribution[rank + 1];

  //

  struct EdgeMessage {
    GlobalNodeID u;
    GlobalNodeID v;
    EdgeWeight weight;
  };
  std::vector<std::vector<EdgeMessage>> out_messages(size);

  for (const auto &[he, weight] : h_graph.edges) {
    const PEID u_owner = he.u / coarse_nodes_per_pe;
    const PEID v_owner = he.v / coarse_nodes_per_pe;

    out_messages[u_owner].emplace_back(he.u, he.v, weight);
    out_messages[v_owner].emplace_back(he.v, he.u, weight);

    DBG << "Send to " << u_owner << ": " << he.u << " --> " << he.v << " / " << weight;
    DBG << "Send to " << v_owner << ": " << he.v << " --> " << he.u << " / " << weight;
  }

  HashedGraph h_c_graph;
  mpi::sparse_alltoall<EdgeMessage>(
      out_messages,
      [&](const auto &buffer) {
        for (const auto &message : buffer) {
          h_c_graph.edges[{message.u, message.v}] += message.weight;
        }
      },
      graph.communicator(), true);

  // construct coarse graph
  const auto from = std::min<GlobalNodeID>(rank * coarse_nodes_per_pe, c_global_n);
  const auto to = std::min<GlobalNodeID>((rank + 1) * coarse_nodes_per_pe, c_global_n);
  ASSERT(from <= to);
  const auto c_n = to - from;

  std::vector<std::vector<std::pair<GlobalNodeID, NodeWeight>>> sorted_graph;
  sorted_graph.resize(c_n);

  EdgeID next_edge_id = 0;
  for (const auto &[he, weight] : h_c_graph.edges) {
    if (from <= he.v && he.v < to) {
      ASSERT(he.u - from < c_n);
      ASSERT(he.v - from < c_n);

      std::pair<GlobalNodeID, NodeWeight> e{he.v, weight / 4};
      sorted_graph[he.u - from].push_back(e);
      DBG << he.u << " -L-> " << he.v << " / " << weight / 4;
      next_edge_id += 1;

      std::pair<GlobalNodeID, NodeWeight> e_rev{he.u, weight / 4};
      sorted_graph[he.v - from].push_back(e_rev);
      DBG << he.v << " -L-> " << he.u << " / " << weight / 4;
      next_edge_id += 1;
    } else {
      ASSERT(he.u - from < c_n) << V(he.u) << V(he.v) << V(from) << V(c_n);

      std::pair<GlobalNodeID, NodeWeight> e{he.v, weight / 2};
      sorted_graph[he.u - from].push_back(e);
      DBG << he.u << " -G-> " << he.v << " / " << weight / 2;
      next_edge_id += 1;
    }
  }

  //  const EdgeID c_m = next_edge_id;
  const auto c_global_m = mpi::allreduce<GlobalEdgeID>(next_edge_id, MPI_SUM, graph.communicator());

  // now construct the graph
  graph::Builder builder;
  builder.initialize(c_global_n, c_global_m, rank, compute_coarse_node_distribution(graph, c_n));

  for (NodeID u = 0; u < c_n; ++u) {
    builder.create_node(0);

    for (EdgeID e = 0; e < sorted_graph[u].size(); e++) {
      const auto v = sorted_graph[u][e].first;
      const auto weight = sorted_graph[u][e].second;
      builder.create_edge(weight, v);
    }
  }

  struct NodeWeightMessage {
    GlobalNodeID node;
    GlobalNodeWeight weight;
  };
  std::vector<std::vector<NodeWeightMessage>> node_weight_out_messages(size);

  for (const auto &[node, weight] : h_graph.nodes) {
    DBG << "Send node weight to " << node / coarse_nodes_per_pe << ": " << node << " := " << weight;
    node_weight_out_messages[node / coarse_nodes_per_pe].emplace_back(node, weight);
  }
  mpi::sparse_alltoall<NodeWeightMessage>(
      node_weight_out_messages,
      [&](const auto buffer) {
        DBG << "got buf size " << buffer.size();
        for (const auto &[node, weight] : buffer) {
          DBG << "Set node weight of " << node - from << " to " << weight;
          builder.change_local_node_weight(node - from, weight);
        }
      },
      graph.communicator(), true);

  return builder.finalize();
}

void update_ghost_node_weights(DistributedGraph &graph) {
  struct Message {
    GlobalNodeID global_node;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message, std::vector>(
      graph,
      [&](const NodeID u) -> Message {
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
  exchange_ghost_node_mapping(graph, mapping, clustering);

  // build coarse graph
  auto h_graph = hash_local_graph(graph, mapping);
  auto c_graph = build_coarse_graph(graph, h_graph, c_global_n);
  update_ghost_node_weights(c_graph);

  return {std::move(c_graph), std::move(mapping)};
}
} // namespace dkaminpar::coarsening