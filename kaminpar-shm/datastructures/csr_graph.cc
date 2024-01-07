/*******************************************************************************
 * Static uncompressed CSR graph data structure.
 *
 * @file:   csr_graph.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/datastructures/csr_graph.h"

#include <kassert/kassert.hpp>

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::shm {
CSRGraph::CSRGraph(
    StaticArray<EdgeID> nodes,
    StaticArray<NodeID> edges,
    StaticArray<NodeWeight> node_weights,
    StaticArray<EdgeWeight> edge_weights,
    const bool sorted
)
    : _nodes(std::move(nodes)),
      _edges(std::move(edges)),
      _node_weights(std::move(node_weights)),
      _edge_weights(std::move(edge_weights)),
      _sorted(sorted) {
  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight = parallel::accumulate(_node_weights, static_cast<NodeWeight>(0));
    _max_node_weight = parallel::max_element(_node_weights);
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight = parallel::accumulate(_edge_weights, static_cast<EdgeWeight>(0));
  }

  _max_degree = parallel::max_difference(_nodes.begin(), _nodes.end());

  init_degree_buckets();
}

CSRGraph::CSRGraph(
    seq,
    StaticArray<EdgeID> nodes,
    StaticArray<NodeID> edges,
    StaticArray<NodeWeight> node_weights,
    StaticArray<EdgeWeight> edge_weights,
    const bool sorted
)
    : _nodes(std::move(nodes)),
      _edges(std::move(edges)),
      _node_weights(std::move(node_weights)),
      _edge_weights(std::move(edge_weights)),
      _sorted(sorted) {
  if (_node_weights.empty()) {
    _total_node_weight = static_cast<NodeWeight>(n());
    _max_node_weight = 1;
  } else {
    _total_node_weight =
        std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }

  if (_edge_weights.empty()) {
    _total_edge_weight = static_cast<EdgeWeight>(m());
  } else {
    _total_edge_weight =
        std::accumulate(_edge_weights.begin(), _edge_weights.end(), static_cast<EdgeWeight>(0));
  }

  init_degree_buckets();
}

void CSRGraph::init_degree_buckets() {
  KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

  if (_sorted) {
    for (const NodeID u : nodes()) {
      ++_buckets[degree_bucket(degree(u)) + 1];
    }
    auto last_nonempty_bucket =
        std::find_if(_buckets.rbegin(), _buckets.rend(), [](const auto n) { return n > 0; });
    _number_of_buckets = std::distance(_buckets.begin(), (last_nonempty_bucket + 1).base());
  } else {
    _buckets[1] = n();
    _number_of_buckets = 1;
  }

  std::partial_sum(_buckets.begin(), _buckets.end(), _buckets.begin());
}

void CSRGraph::update_total_node_weight() {
  if (_node_weights.empty()) {
    _total_node_weight = n();
    _max_node_weight = 1;
  } else {
    _total_node_weight =
        std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
    _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
  }
}

void CSRGraph::update_degree_buckets() {
  std::fill(_buckets.begin(), _buckets.end(), 0);
  init_degree_buckets();
}

} // namespace kaminpar::shm
