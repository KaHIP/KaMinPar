#pragma once

#include "definitions.h"
#include "../simple_graph.h"

#include <ranges>

namespace kaminpar::tool::converter {
class TwoPassGraphBuilder {
  static constexpr auto kDebug = false;

public:
  explicit TwoPassGraphBuilder(const NodeID n) : _nodes(n + 1), _node_weights(n, 1) {}

  TwoPassGraphBuilder(const TwoPassGraphBuilder &) = delete;
  TwoPassGraphBuilder &operator=(const TwoPassGraphBuilder &) = delete;

  TwoPassGraphBuilder(TwoPassGraphBuilder &&) noexcept = default;
  TwoPassGraphBuilder &operator=(TwoPassGraphBuilder &&) noexcept = default;

  void pass1_add_edge(const NodeID u, const NodeID v) {
    ASSERT(u < v) << "smaller node id should go first";
    ASSERT(u < _node_weights.size());
    ASSERT(v < _node_weights.size());
    ++_nodes[u];
    ++_nodes[v];
    _m += 2;

    DBG << "Add edge " << u << " --> " << v;
  }

  void pass1_finish() {
    _edges.resize(_m);
    _edge_weights.resize(_m);

    EdgeID prefix_sum = 0;
    for (std::size_t i = 0; i < _nodes.size(); ++i) {
      const EdgeID degree = _nodes[i];
      _nodes[i] = prefix_sum;
      prefix_sum += degree;
    }

    DBG << _nodes;
  }

  void pass2_add_edge(const NodeID u, const NodeID v, const EdgeWeight weight = 1) {
    DBG << "Add edge " << u << " --> " << v << " with weight " << weight;

    _edges[_nodes[u]] = v;
    _edge_weights[_nodes[u]] = weight;
    _edges[_nodes[v]] = u;
    _edge_weights[_nodes[v]] = weight;
    ++_nodes[u];
    ++_nodes[v];
  }

  SimpleGraph pass2_finish() {
    for (std::size_t i = _nodes.size() - 2; i > 0; --i) { _nodes[i] = _nodes[i - 1]; }
    _nodes[0] = 0;

    bool has_node_weights = std::ranges::any_of(_node_weights, [](const NodeWeight &weight) { return weight != 1; });
    bool has_edge_weights = std::ranges::any_of(_edge_weights, [](const EdgeWeight &weight) { return weight != 1; });

    SimpleGraph graph{.nodes = std::move(_nodes), .edges = std::move(_edges), .node_weights = {}, .edge_weights = {}};
    if (has_node_weights) { graph.node_weights = std::move(_node_weights); }
    if (has_edge_weights) { graph.edge_weights = std::move(_edge_weights); }

    return graph;
  }

  void set_node_weight(const NodeID u, const NodeWeight weight) {
    ASSERT(u < _node_weights.size());
    _node_weights[u] = weight;
  }

private:
  std::vector<EdgeID> _nodes;
  std::vector<NodeID> _edges{};
  std::vector<NodeWeight> _node_weights;
  std::vector<EdgeWeight> _edge_weights{};
  EdgeID _m{0};
};
} // namespace kaminpar::tool::converter