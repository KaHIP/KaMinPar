/*******************************************************************************
 * @file:   seq_global_clustering_contraction_redistribution.cc
 *
 * @author: Daniel Seemaier
 * @date:   26.10.2021
 * @brief:  Sequential code to contract a global clustering without any
 * limitations and redistribute the contracted graph such that each PE gets
 * an equal number of edges.
 ******************************************************************************/
#include "dkaminpar/coarsening/seq_global_clustering_contraction_redistribution.h"

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/mpi_wrapper.h"

#include <vector>

namespace dkaminpar::coarsening {
SET_DEBUG(false);

/*
 * Global cluster contraction with complete graph redistribution
 */

namespace {
// global mapping, global number of coarse nodes
std::pair<std::unordered_map<GlobalNodeID, GlobalNodeID>, GlobalNodeID>
compute_mapping(const DistributedGraph &graph,
                const scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>> &clustering) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());
  const GlobalNodeID fine_nodes_per_pe = std::ceil(1.0 * graph.global_n() / size);

  // mark global node ids that are used as cluster ids
  using ConcurrentFilter = std::unordered_map<GlobalNodeID, bool>;
  std::vector<ConcurrentFilter> filters(size);
  std::vector<std::vector<GlobalNodeID>> used_ids_messages(size);
  for (NodeID u : graph.nodes()) {
    const auto u_cluster = clustering[u];
    const PEID new_owner = static_cast<PEID>(u_cluster / fine_nodes_per_pe);
    if (!filters[new_owner][u_cluster]) {
      filters[new_owner][u_cluster] = true;
      used_ids_messages[new_owner].push_back(u_cluster);
    }
  }

  // exchange messages -- store incoming messages for reply
  std::vector<GlobalNodeID> local_labels; // all labels on our PE
  auto inc_messages = mpi::sparse_alltoall_get<GlobalNodeID, std::vector>(used_ids_messages, graph.communicator(), true);
  for (const auto &inc_messages_from_pe : inc_messages) {
    for (const auto &node : inc_messages_from_pe) {
      local_labels.push_back(node);
    }
  }

  // filter duplicates
  std::sort(local_labels.begin(), local_labels.end());
  auto it = std::unique(local_labels.begin(), local_labels.end());
  local_labels.resize(std::distance(local_labels.begin(), it));

  const GlobalNodeID local_n = local_labels.size();
  const GlobalNodeID prefix_n = mpi::scan(local_n, MPI_SUM, graph.communicator());
  const GlobalNodeID c_global_n = mpi::bcast(prefix_n, size - 1, graph.communicator());
  const GlobalNodeID offset_n = prefix_n - local_n;
  DBG << V(local_labels) << V(local_n) << V(prefix_n) << V(c_global_n) << V(offset_n);

  // build mapping
  std::unordered_map<GlobalNodeID, GlobalNodeID> label_mapping;

  GlobalNodeID cur_id = offset_n;
  for (const auto &node : local_labels) {
    label_mapping[node] = cur_id++;
  }

  // send mappings back
  std::vector<std::vector<GlobalNodeID>> out_messages(size);
  for (PEID pe = 0; pe < size; ++pe) {
    for (const auto &node : inc_messages[pe]) {
      out_messages[pe].push_back(label_mapping[node]);
    }
  }
  label_mapping.clear();

  mpi::sparse_alltoall<GlobalNodeID>(
      out_messages,
      [&](const auto &buffer, const PEID pe) {
        for (std::size_t i = 0; i < buffer.size(); ++i) {
          label_mapping[used_ids_messages[pe][i]] = buffer[i];
        }
      },
      graph.communicator(), true);

  return {label_mapping, c_global_n};
}

void exchange_ghost_node_mapping(const DistributedGraph &graph, auto &label_mapping, auto &clustering) {
  struct Message {
    GlobalNodeID global_node;
    GlobalNodeID coarse_global_node;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message, std::vector>(
      graph,
      [&](const NodeID u) -> Message {
        const GlobalNodeID global_u = graph.local_to_global_node(u);
        return {global_u, label_mapping[clustering[u]]};
      },
      [&](const auto buffer) {
        for (const Message &message : buffer) {
          label_mapping[message.global_node] = message.coarse_global_node;
        }
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

scalable_vector<GlobalNodeID> compute_coarse_node_distribution(const DistributedGraph &graph, const NodeID c_n) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());

  // Compute new node distribution, total number of coarse nodes
  const GlobalNodeID last_node = mpi::scan(static_cast<GlobalNodeID>(c_n), MPI_SUM, graph.communicator());
  [[maybe_unused]] const GlobalNodeID first_node = last_node - c_n;
  scalable_vector<GlobalNodeID> c_node_distribution(size + 1);
  c_node_distribution[rank + 1] = last_node;
  mpi::allgather(&c_node_distribution[rank + 1], 1, c_node_distribution.data() + 1, 1, graph.communicator());

  return c_node_distribution;
}

DistributedGraph build_coarse_graph(const DistributedGraph &graph, HashedGraph &h_graph,
                                    const GlobalNodeID c_global_n) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());
  const GlobalNodeID coarse_nodes_per_pe = std::ceil(1.0 * c_global_n / size);

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
        for (const Message &message : buffer) {
          const NodeID local_node = graph.global_to_local_node(message.global_node);
          graph.set_ghost_node_weight(local_node, message.weight);
        }
      });
}
} // namespace

contraction::GlobalMappingResult contract_global_clustering_redistribute_sequential(
    const DistributedGraph &graph,
    const scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>> &clustering,
    contraction::MemoryContext m_ctx) {
  // compute local mapping for owned nodes
  auto [global_mapping, c_global_n] = compute_mapping(graph, clustering);
  scalable_vector<GlobalNodeID> local_mapping(graph.total_n());
  for (const NodeID u : graph.nodes()) {
    local_mapping[u] = global_mapping[clustering[u]];
  }

  // compute local mapping for ghost nodes
  exchange_ghost_node_mapping(graph, global_mapping, clustering);
  for (const NodeID u : graph.ghost_nodes()) {
    local_mapping[u] = global_mapping[graph.local_to_global_node(u)];
  }

  // build coarse graph
  auto h_graph = hash_local_graph(graph, local_mapping);
  auto c_graph = build_coarse_graph(graph, h_graph, c_global_n);
  update_ghost_node_weights(c_graph);

  return {std::move(c_graph), std::move(local_mapping), std::move(m_ctx)};
}
} // namespace dkaminpar::coarsening