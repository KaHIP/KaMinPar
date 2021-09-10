#include "dkaminpar/algorithm/seq_global_graph_contraction_redistribution.h"

#include "dkaminpar/mpi_graph_utils.h"
#include "dkaminpar/mpi_utils.h"

#include <vector>

namespace dkaminpar::graph {
/*
 * Global cluster contraction with complete graph redistribution
 */

namespace {
// global mapping, global number of coarse nodes
std::pair<std::unordered_map<GlobalNodeID, GlobalNodeID>, GlobalNodeID>
compute_mapping(const DistributedGraph &graph, const scalable_vector<GlobalNodeID> &clustering) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());
  const GlobalNodeID fine_nodes_per_pe = graph.global_n() / size;

  // mark global node ids that are used as cluster ids
  using ConcurrentFilter = std::unordered_map<GlobalNodeID, bool>;
  std::vector<ConcurrentFilter> filters(size);
  std::vector<std::vector<GlobalNodeID>> used_ids_messages(size);
  for (NodeID u : graph.nodes()) {
    const PEID new_owner = static_cast<PEID>(clustering[u] / fine_nodes_per_pe);
    const auto cluster_u = clustering[u];
    if (!filters[new_owner][cluster_u]) {
      filters[new_owner][cluster_u] = true;
      used_ids_messages[new_owner].push_back(u);
    }
  }

  // exchange messages -- store incoming messages for reply
  std::vector<GlobalNodeID> local_labels; // all labels on our PE
  auto inc_messages = mpi::sparse_all_to_all_get<std::vector>(used_ids_messages, 0, graph.communicator(), true);
  for (const auto &inc_messages_from_pe : inc_messages) {
    for (const auto &node : inc_messages_from_pe) { local_labels.push_back(node); }
  }

  // filter duplicates
  std::sort(local_labels.begin(), local_labels.end());
  auto it = std::unique(local_labels.begin(), local_labels.end());
  local_labels.resize(std::distance(local_labels.begin(), it));

  const GlobalNodeID local_n = local_labels.size();
  const GlobalNodeID prefix_n = mpi::scan(local_n, MPI_SUM, graph.communicator());
  const GlobalNodeID c_n = mpi::bcast(prefix_n, size - 1, graph.communicator());
  const GlobalNodeID offset_n = prefix_n - local_n;

  // build mapping
  std::unordered_map<GlobalNodeID, GlobalNodeID> label_mapping;

  GlobalNodeID cur_id = offset_n;
  for (const auto &node : local_labels) { label_mapping[node] = cur_id++; }

  // send mappings back
  std::vector<std::vector<GlobalNodeID>> out_messages(size);
  for (PEID pe = 0; pe < size; ++pe) {
    for (const auto &node : inc_messages[pe]) { out_messages[pe].push_back(label_mapping[node]); }
  }
  label_mapping.clear();

  mpi::sparse_all_to_all<std::vector>(
      out_messages, 0,
      [&](const PEID pe, const auto &buffer) {
        for (std::size_t i = 0; i < buffer.size(); ++i) { label_mapping[used_ids_messages[pe][i]] = buffer[i]; }
      },
      graph.communicator(), true);

  return {label_mapping, c_n};
}

void exchange_ghost_node_mapping(const DistributedGraph &graph, auto &label_mapping) {
  struct Message {
    GlobalNodeID global_node;
    GlobalNodeID coarse_global_node;
  };

  mpi::graph::sparse_alltoall_interface_node<Message, std::vector>(
      graph,
      [&](const NodeID u, const PEID /* pe */) -> Message {
        const GlobalNodeID global_u = graph.local_to_global_node(u);
        return {global_u, label_mapping[global_u]};
      },
      [&](const PEID /* pe */, const auto &buffer) {
        for (const Message &message : buffer) { label_mapping[message.global_node] = message.coarse_global_node; }
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
    const GlobalNodeID c_u = label_mapping[graph.local_to_global_node(u)];
    h_graph.nodes[c_u] += graph.node_weight(u);

    for (const auto [e, v] : graph.neighbors(u)) {
      const GlobalNodeID c_v = label_mapping[graph.local_to_global_node(v)];
      if (c_v != c_u) { h_graph.edges[{c_u, c_v}] += graph.edge_weight(e); }
    }
  }

  return h_graph;
}

DistributedGraph build_coarse_graph(const DistributedGraph &graph, HashedGraph &h_graph, const GlobalNodeID c_n) {
  const auto [size, rank] = mpi::get_comm_info(graph.communicator());
  const GlobalNodeID coarse_nodes_per_pe = c_n / size;

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
  }

  HashedGraph h_c_graph;
  mpi::sparse_all_to_all<std::vector>(
      out_messages, 0,
      [&](const PEID /* pe */, const auto &buffer) {
        for (const auto &message : buffer) { h_c_graph.edges[{message.u, message.v}] += message.weight; }
      },
      graph.communicator(), true);

  return {};
}

void update_ghost_node_weights(DistributedGraph &graph) {
  struct Message {
    GlobalNodeID global_node;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_interface_node<Message, std::vector>(
      graph,
      [&](const NodeID u, const PEID /* pe */) -> Message {
        return {graph.local_to_global_node(u), graph.node_weight(u)};
      },
      [&](const PEID /* pe */, const auto &buffer) {
        for (const Message &message : buffer) {
          const NodeID local_node = graph.global_to_local_node(message.global_node);
          graph.set_ghost_node_weight(local_node, message.weight);
        }
      });
}

scalable_vector<GlobalNodeID> build_local_mapping(const DistributedGraph &graph, auto &global_mapping) {
  scalable_vector<GlobalNodeID> local_mapping(graph.total_n());
  for (const NodeID u : graph.all_nodes()) { local_mapping[u] = global_mapping[graph.local_to_global_node(u)]; }
  return local_mapping;
}
} // namespace

contraction::GlobalMappingResult
contract_global_clustering_redistribute(const DistributedGraph &graph, const scalable_vector<GlobalNodeID> &clustering,
                                        contraction::MemoryContext m_ctx) {
  auto [global_mapping, c_n] = compute_mapping(graph, clustering);
  exchange_ghost_node_mapping(graph, global_mapping);
  auto h_graph = hash_local_graph(graph, global_mapping);
  auto c_graph = build_coarse_graph(graph, h_graph, c_n);
  update_ghost_node_weights(c_graph);
  auto local_mapping = build_local_mapping(graph, global_mapping);

  return {std::move(c_graph), std::move(local_mapping), std::move(m_ctx)};
}
} // namespace dkaminpar::graph