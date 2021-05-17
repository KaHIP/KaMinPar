#pragma once

#include "datastructure/graph.h"
#include "definitions.h"

#include <ranges>

namespace kaminpar::tool {
struct SimpleGraph {
  std::vector<EdgeID> nodes;
  std::vector<NodeID> edges;
  std::vector<NodeWeight> node_weights;
  std::vector<EdgeWeight> edge_weights;

  [[nodiscard]] NodeID n() const { return nodes.size() - 1; }
  [[nodiscard]] EdgeID m() const { return edges.size(); }
  [[nodiscard]] auto nodes_iter() const { return std::ranges::views::iota(static_cast<NodeID>(0), nodes.size() - 1); }
  [[nodiscard]] auto edges_iter() const { return std::ranges::views::iota(static_cast<EdgeID>(0), edges.size()); }
  [[nodiscard]] auto neighbors_iter(const NodeID u) const {
    return std::views::iota(nodes[u], nodes[u + 1]) |
           std::views::transform([this](const EdgeID e) { return std::make_pair(e, this->edges[e]); });
  }
  [[nodiscard]] Degree degree(const NodeID u) const { return nodes[u + 1] - nodes[u]; }
  [[nodiscard]] bool has_node_weights() const { return !node_weights.empty(); }
  [[nodiscard]] bool has_edge_weights() const { return !edge_weights.empty(); }
};

SimpleGraph graph_to_simple_graph(Graph graph) {
  std::vector<EdgeID> nodes(graph.n() + 1);
  std::vector<NodeID> edges(graph.m());
  std::vector<NodeWeight> node_weights(graph.is_node_weighted() * graph.n());
  std::vector<EdgeWeight> edge_weights(graph.is_edge_weighted() * graph.m());

  std::ranges::copy(graph.raw_nodes(), nodes.begin());
  std::ranges::copy(graph.raw_edges(), edges.begin());
  std::ranges::copy(graph.raw_node_weights(), node_weights.begin());
  std::ranges::copy(graph.raw_edge_weights(), edge_weights.begin());

  return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
}

Graph simple_graph_to_graph(SimpleGraph graph) {
  StaticArray<EdgeID> nodes(graph.n() + 1);
  StaticArray<NodeID> edges(graph.m());

  std::ranges::copy(graph.nodes, nodes.begin());
  std::ranges::copy(graph.edges, edges.begin());

  if (graph.has_node_weights() || graph.has_edge_weights()) {
    StaticArray<NodeWeight> node_weights(graph.n(), 1);
    StaticArray<EdgeWeight> edge_weights(graph.m(), 1);
    if (graph.has_node_weights()) { std::ranges::copy(graph.node_weights, node_weights.begin()); }
    if (graph.has_edge_weights()) { std::ranges::copy(graph.edge_weights, edge_weights.begin()); }
    return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
  }

  return {std::move(nodes), std::move(edges)};
}
} // namespace kaminpar::tool