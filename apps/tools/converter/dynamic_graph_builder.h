#pragma once

namespace kaminpar::tool::converter {
class DynamicGraphBuilder {
public:
  explicit DynamicGraphBuilder(const NodeID n = 0) : _adjacency_matrix(n), _node_weights(n, 1), _n(n), _m(0) {}

  DynamicGraphBuilder(const DynamicGraphBuilder &) = delete;
  DynamicGraphBuilder &operator=(const DynamicGraphBuilder &) = delete;

  DynamicGraphBuilder(DynamicGraphBuilder &&) noexcept = default;
  DynamicGraphBuilder &operator=(DynamicGraphBuilder &&) noexcept = default;

  void set_node_weight(const NodeID u, const NodeWeight weight) {
    ensure_size(u);
    _node_weights[u] = weight;
  }

  template<bool add_reverse = false>
  void add_edge(const NodeID u, const NodeID v, const EdgeWeight weight) {
    ensure_size(u);
    if (_adjacency_matrix[u][v] == 0) ++_m;
    _adjacency_matrix[u][v] += weight;
    if constexpr (add_reverse) {
      ensure_size(v);
      if (_adjacency_matrix[v][u] == 0) ++_m;
      _adjacency_matrix[v][u] += weight;
    }
  }

  [[nodiscard]] SimpleGraph build() const {
    std::vector<EdgeID> nodes;
    std::vector<NodeID> edges;
    std::vector<EdgeWeight> edge_weights;
    nodes.reserve(_n + 1);
    edges.reserve(_m);
    ASSERT(_node_weights.size() == _n);
    edge_weights.reserve(_m);

    nodes.push_back(0);
    for (NodeID u = 0; u < _n; ++u) {
      for (const auto &[v, weight] : _adjacency_matrix[u]) {
        edges.push_back(v);
        edge_weights.push_back(weight);
      }
      nodes.push_back(edges.size());
    }

    return SimpleGraph{.nodes = std::move(nodes),
                       .edges = std::move(edges),
                       .node_weights = _node_weights,
                       .edge_weights = std::move(edge_weights)};
  }

private:
  void ensure_size(const NodeID u) {
    _n = std::max(_n, u + 1);
    for (NodeID u_prime = _adjacency_matrix.size(); u_prime <= u; ++u_prime) {
      _adjacency_matrix.emplace_back();
      _node_weights.push_back(1);
    }
  }

  std::vector<std::unordered_map<NodeID, EdgeWeight>> _adjacency_matrix;
  std::vector<NodeWeight> _node_weights;
  NodeID _n;
  EdgeID _m;
};
} // namespace kaminpar::tool::converter