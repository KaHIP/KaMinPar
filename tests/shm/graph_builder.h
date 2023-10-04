#pragma once

#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm::testing {
class GraphBuilder {
public:
  GraphBuilder() = default;

  GraphBuilder(const NodeID n, const EdgeID m) {
    _nodes.reserve(n + 1);
    _edges.reserve(m);
    _node_weights.reserve(n);
    _edge_weights.reserve(m);
  }

  GraphBuilder(const GraphBuilder &) = delete;
  GraphBuilder &operator=(const GraphBuilder &) = delete;

  GraphBuilder(GraphBuilder &&) noexcept = default;
  GraphBuilder &operator=(GraphBuilder &&) noexcept = default;

  NodeID new_node(const NodeWeight weight = 1) {
    _nodes.push_back(_edges.size());
    _node_weights.push_back(weight);
    return _nodes.size() - 1;
  }

  NodeWeight &last_node_weight() {
    return _node_weights.back();
  }

  EdgeID new_edge(const NodeID v, const EdgeID weight = 1) {
    _edges.push_back(v);
    _edge_weights.push_back(weight);
    return _edges.size() - 1;
  }

  EdgeWeight &last_edge_weight() {
    return _edge_weights.back();
  }

  template <typename... Args> Graph build(Args &&...args) {
    _nodes.push_back(_edges.size());
    return Graph(
        static_array::create_from(_nodes),
        static_array::create_from(_edges),
        static_array::create_from(_node_weights),
        static_array::create_from(_edge_weights),
        std::forward<Args>(args)...
    );
  }

private:
  std::vector<EdgeID> _nodes{};
  std::vector<NodeID> _edges{};
  std::vector<NodeWeight> _node_weights{};
  std::vector<EdgeWeight> _edge_weights{};
};

/*!
 * Builds a single graph that contains all graphs as induced subgraphs.
 *
 * @param graphs A list of graphs that should be copied.
 * @param connect_graphs If true, the first node of each graph is connected to a
 * clique with edges of weight 1. Otherwise, the induced subgraphs are
 * disconnected.
 * @return A single graph containing all other graphs.
 */
inline Graph
merge_graphs(std::initializer_list<Graph *> graphs, const bool connect_graphs = false) {
  const NodeID n =
      std::accumulate(graphs.begin(), graphs.end(), 0, [&](const NodeID acc, const Graph *graph) {
        return acc + graph->n();
      });
  const EdgeID m =
      std::accumulate(graphs.begin(), graphs.end(), 0, [&](const EdgeID acc, const Graph *graph) {
        return acc + graph->m();
      });
  GraphBuilder builder(n, m);

  NodeID offset = 0;
  for (const Graph *graph : graphs) {
    KASSERT(graph->n() > 0u);
    builder.new_node(graph->node_weight(0));

    if (connect_graphs) {
      NodeID first_node_in_other_graph = 0;
      for (const Graph *other_graph : graphs) {
        if (other_graph == graph) {
          continue;
        }
        KASSERT(other_graph->n() > 0u);
        builder.new_edge(first_node_in_other_graph);
        first_node_in_other_graph += other_graph->n();
      }
    }

    for (const NodeID u : graph->nodes()) {
      if (u > 0) {
        builder.new_node(graph->node_weight(u));
      }
      for (const auto [e, v] : graph->neighbors(u)) {
        builder.new_edge(offset + v, graph->edge_weight(e));
      }
    }

    offset += graph->n();
  }

  return builder.build();
}
} // namespace kaminpar::shm::testing
