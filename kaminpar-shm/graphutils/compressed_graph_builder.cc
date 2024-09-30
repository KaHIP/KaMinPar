/*******************************************************************************
 * Sequential builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.cc
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

#include <utility>

namespace kaminpar::shm {

CompressedGraph CompressedGraphBuilder::compress(const CSRGraph &graph) {
  const bool store_node_weights = graph.is_node_weighted();
  const bool store_edge_weights = graph.is_edge_weighted();

  CompressedGraphBuilder builder(
      graph.n(), graph.m(), store_node_weights, store_edge_weights, graph.sorted()
  );

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  neighbourhood.reserve(graph.max_degree());

  for (const NodeID u : graph.nodes()) {
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      neighbourhood.emplace_back(v, w);
    });

    builder.add_node(u, neighbourhood);
    if (store_node_weights) {
      builder.add_node_weight(u, graph.node_weight(u));
    }

    neighbourhood.clear();
  }

  return builder.build();
}

CompressedGraphBuilder::CompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _sorted(sorted),
      _compressed_neighborhoods_builder(num_nodes, num_edges, has_edge_weights),
      _store_node_weights(has_node_weights),
      _total_node_weight(0) {
  if (has_node_weights) {
    _node_weights.resize(num_nodes, static_array::noinit);
  }
}

void CompressedGraphBuilder::add_node(
    const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  _compressed_neighborhoods_builder.add(node, neighbourhood);
}

void CompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  KASSERT(_store_node_weights);

  _total_node_weight += weight;
  _node_weights[node] = weight;
}

CompressedGraph CompressedGraphBuilder::build() {
  CompressedNeighborhoods compressed_neighborhoods = _compressed_neighborhoods_builder.build();

  const bool unit_node_weights =
      std::cmp_equal(_total_node_weight + 1, compressed_neighborhoods.num_nodes());
  if (unit_node_weights) {
    _node_weights.free();
  }

  return CompressedGraph(std::move(compressed_neighborhoods), std::move(_node_weights), _sorted);
}

std::size_t CompressedGraphBuilder::currently_used_memory() const {
  return _compressed_neighborhoods_builder.currently_used_memory() +
         _node_weights.size() * sizeof(NodeWeight);
}

std::int64_t CompressedGraphBuilder::total_node_weight() const {
  return _total_node_weight;
}

std::int64_t CompressedGraphBuilder::total_edge_weight() const {
  return _compressed_neighborhoods_builder.total_edge_weight();
}

} // namespace kaminpar::shm
