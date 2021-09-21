#pragma once

#include "kaminpar/datastructure/graph.h"

namespace kaminpar {
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

  NodeWeight &last_node_weight() { return _node_weights.back(); }

  EdgeID new_edge(const NodeID v, const EdgeID weight = 1) {
    _edges.push_back(v);
    _edge_weights.push_back(weight);
    return _edges.size() - 1;
  }

  EdgeWeight &last_edge_weight() { return _edge_weights.back(); }

  template<typename... Args>
  Graph build(Args &&...args) {
    _nodes.push_back(_edges.size());
    return Graph(from_vec(_nodes), from_vec(_edges), from_vec(_node_weights), from_vec(_edge_weights),
                 std::forward<Args>(args)...);
  }

private:
  std::vector<EdgeID> _nodes{};
  std::vector<NodeID> _edges{};
  std::vector<NodeWeight> _node_weights{};
  std::vector<EdgeWeight> _edge_weights{};
};
} // namespace kaminpar