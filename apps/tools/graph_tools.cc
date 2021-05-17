#include "graph_tools.h"

#include "datastructure/queue.h"
#include "utility/console_io.h"

namespace kaminpar::tool {
std::vector<NodeWeight> compute_k_core(const Graph &graph, const EdgeWeight k, std::vector<EdgeWeight> core) {
  ALWAYS_ASSERT(k > 0);
  ALWAYS_ASSERT(core.empty() || core.size() == graph.n());

  namespace ranges = std::ranges;

  if (core.empty()) {
    core.resize(graph.n());
    ranges::transform(graph.nodes(), core.begin(), [&graph](const NodeID u) {
      const auto &incident_edge_weights = graph.incident_edges(u) | ranges::views::transform([&graph](const EdgeID e) {
                                            return graph.edge_weight(e);
                                          });
      return std::accumulate(incident_edge_weights.begin(), incident_edge_weights.end(), static_cast<EdgeWeight>(0));
    });
  }

  std::vector<bool> in_queue(graph.n(), true);
  std::vector<bool> todo_delete(graph.n());
  std::deque<NodeID> queue(graph.n());
  std::iota(queue.begin(), queue.end(), 0);

  while (!queue.empty()) {
    const NodeID u = queue.front();
    queue.pop_front();
    in_queue[u] = false;

    if ((0 < core[u] && core[u] < k) || todo_delete[u]) {
      core[u] = 0;
      todo_delete[u] = false;

      for (const auto [e, v] : graph.neighbors(u)) {
        if (core[v] > 0) {
          core[v] -= graph.edge_weight(e);
          todo_delete[v] = core[v] < k;
          if (todo_delete[v] && !in_queue[v]) {
            queue.push_back(v);
            in_queue[v] = true;
          }
        }
      }
    }
  }

  ASSERT(queue.empty());
  ASSERT(ranges::none_of(in_queue, std::identity{}));
  ASSERT(ranges::none_of(todo_delete, std::identity{}));

  return core;
}

KCoreStatistics compute_k_core_statistics(const Graph &graph, const std::vector<EdgeWeight> &k_core) {
  KCoreStatistics stats{};

  for (const NodeID u : graph.nodes()) {
    if (k_core[u] > 0) {
      stats.k = std::min(stats.k, k_core[u]);
      stats.n += 1;
      stats.max_node_weight = std::max(stats.max_node_weight, graph.node_weight(u));
      stats.total_node_weight += graph.node_weight(u);
      stats.total_edge_weight += k_core[u];
      stats.max_weighted_degree = std::max(stats.max_weighted_degree, k_core[u]);

      Degree degree{0};
      for (const auto [e, v] : graph.neighbors(u)) {
        if (k_core[v] > 0) {
          stats.m += 1;
          stats.max_edge_weight = std::max(stats.max_edge_weight, graph.edge_weight(e));
          ++degree;
        }
      }

      stats.max_degree = std::max(stats.max_degree, degree);
    }
  }

  return stats;
}

std::vector<bool> k_core_to_indicator_array(const std::vector<EdgeWeight> &k_core) {
  std::vector<bool> contained(k_core.size());
  std::transform(k_core.begin(), k_core.end(), contained.begin(), [](const EdgeWeight deg) { return deg > 0; });
  return contained;
}

std::vector<BlockID> find_connected_components(const Graph &graph) {
  std::vector<BlockID> component(graph.n());
  std::vector<bool> marked(graph.n());

  Queue<NodeID> queue(graph.n());
  NodeID first_unmarked_node = 0;

  NodeID num_assigned_nodes = 1;
  NodeID current_component = -1; // is ok

  while (num_assigned_nodes < graph.n()) {
    if (queue.empty()) {
      ASSERT(first_unmarked_node < graph.n());
      queue.push_tail(first_unmarked_node);
      marked[first_unmarked_node] = true;
      while (first_unmarked_node < graph.n() && marked[first_unmarked_node]) { first_unmarked_node++; }
      ++current_component;
    }

    const NodeID u = queue.head();
    queue.pop_head();
    component[u] = current_component;
    ++num_assigned_nodes;

    for (const NodeID v : graph.adjacent_nodes(u)) {
      if (marked[v]) continue;
      queue.push_tail(v);
      marked[v] = true;
      if (v == first_unmarked_node) {
        while (first_unmarked_node < graph.n() && marked[first_unmarked_node]) { first_unmarked_node++; }
      }
    }
  }

  return component;
}
} // namespace kaminpar::tool