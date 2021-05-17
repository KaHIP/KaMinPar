#pragma once

#include "datastructure/graph.h"

namespace kaminpar::tool {
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

  NodeID new_node(NodeWeight weight = 1);
  EdgeID new_edge(NodeID v, EdgeID weight = 1);

  NodeWeight &last_node_weight() { return _node_weights.back(); }
  EdgeWeight &last_edge_weight() { return _edge_weights.back(); }

  template<typename... Args>
  Graph build(Args &&...args) {
    _nodes.push_back(_edges.size());
    return Graph(std::move(_nodes), std::move(_edges), std::move(_node_weights), std::move(_edge_weights),
                 std::forward<Args>(args)...);
  }

private:
  std::vector<EdgeID> _nodes;
  std::vector<NodeID> _edges;
  std::vector<NodeWeight> _node_weights;
  std::vector<EdgeWeight> _edge_weights;
};

template<typename... GraphArgs>
Graph complete_bipartite(const NodeID n, const NodeID m, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) { // set A
    builder.new_node();
    for (NodeID v = n; v < n + m; ++v) { builder.new_edge(v); }
  }
  for (NodeID u = n; u < n + m; ++u) { // set B
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) { builder.new_edge(v); }
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

template<typename... GraphArgs>
Graph star(const NodeID n, GraphArgs &&...graph_args) {
  return complete_bipartite(1, n, std::forward<GraphArgs...>(graph_args)...);
}

template<typename... GraphArgs>
Graph empty(const NodeID n, GraphArgs &&...graph_args) {
  GraphBuilder builder(n, 0);
  for (NodeID u = 0; u < n; ++u) { builder.new_node(); }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}
} // namespace kaminpar::tool