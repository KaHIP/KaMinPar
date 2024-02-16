/*******************************************************************************
 * Static graph with CSR representation.
 *
 * @file:   graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <numeric>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/tags.h"

namespace kaminpar::shm {
class Graph {
public:
  // Data types used by this graph
  using NodeID = ::kaminpar::shm::NodeID;
  using NodeWeight = ::kaminpar::shm::NodeWeight;
  using EdgeID = ::kaminpar::shm::EdgeID;
  using EdgeWeight = ::kaminpar::shm::EdgeWeight;

  // Tag for the sequential ctor.
  struct seq {};

  Graph() = default;

  Graph(
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  Graph(
      seq,
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;

  //
  // Access to raw data members
  //

  [[nodiscard]] inline StaticArray<EdgeID> &raw_nodes() {
    return _nodes;
  }

  [[nodiscard]] inline const StaticArray<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  [[nodiscard]] inline StaticArray<NodeID> &raw_edges() {
    return _edges;
  }

  [[nodiscard]] inline const StaticArray<NodeID> &raw_edges() const {
    return _edges;
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() {
    return _node_weights;
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const {
    return _node_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &raw_edge_weights() {
    return _edge_weights;
  }

  [[nodiscard]] inline const StaticArray<EdgeWeight> &raw_edge_weights() const {
    return _edge_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeID> &&take_raw_nodes() {
    return std::move(_nodes);
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_edges() {
    return std::move(_edges);
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() {
    return std::move(_node_weights);
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &&take_raw_edge_weights() {
    return std::move(_edge_weights);
  }

  //
  // Node weights
  //

  [[nodiscard]] inline bool node_weighted() const {
    return static_cast<NodeWeight>(n()) != total_node_weight();
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const {
    return _total_node_weight;
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const {
    return _max_node_weight;
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const {
    KASSERT(!node_weighted() || u < _node_weights.size());
    return node_weighted() ? _node_weights[u] : 1;
  }

  //
  // Edge weights
  //

  [[nodiscard]] inline bool edge_weighted() const {
    return static_cast<EdgeWeight>(m()) != total_edge_weight();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const {
    return _total_edge_weight;
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const {
    KASSERT(!edge_weighted() || e < _edge_weights.size());
    return edge_weighted() ? _edge_weights[e] : 1;
  }

  //
  // Graph properties
  //

  [[nodiscard]] inline NodeID n() const {
    return static_cast<NodeID>(_nodes.size() - 1);
  }

  [[nodiscard]] inline EdgeID m() const {
    return static_cast<EdgeID>(_edges.size());
  }

  //
  // Low-level graph structure
  //

  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const {
    KASSERT(e < _edges.size());
    return _edges[e];
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return static_cast<NodeID>(_nodes[u + 1] - _nodes[u]);
  }

  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return _nodes[u];
  }

  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return _nodes[u + 1];
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  //
  // Sequential iteration
  //

  [[nodiscard]] inline IotaRange<NodeID> nodes() const {
    return {static_cast<NodeID>(0), n()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const {
    return {static_cast<EdgeID>(0), m()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return {_nodes[u], _nodes[u + 1]};
  }

  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return TransformedIotaRange(_nodes[u], _nodes[u + 1], [this](const EdgeID e) {
      return this->edge_target(e);
    });
  }

  [[nodiscard]] inline auto neighbors(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return TransformedIotaRange(_nodes[u], _nodes[u + 1], [this](const EdgeID e) {
      return std::make_pair(e, this->edge_target(e));
    });
  }

  //
  // Graph permutation
  //

  inline void set_permutation(StaticArray<NodeID> permutation) {
    _permutation = std::move(permutation);
  }

  [[nodiscard]] inline bool permuted() const {
    return !_permutation.empty();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const {
    KASSERT(u < _permutation.size());
    return _permutation[u];
  }

  inline StaticArray<NodeID> &&take_raw_permutation() {
    return std::move(_permutation);
  }

  //
  // Degree buckets
  //

  [[nodiscard]] inline NodeID bucket_size(const std::size_t bucket) const {
    KASSERT(bucket + 1 < _buckets.size());
    return _buckets[bucket + 1] - _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const {
    return _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const {
    return first_node_in_bucket(bucket + 1);
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const {
    return _number_of_buckets;
  }

  [[nodiscard]] inline bool sorted() const {
    return _sorted;
  }

  void update_total_node_weight();

private:
  void init_degree_buckets();

  StaticArray<EdgeID> _nodes;
  StaticArray<NodeID> _edges;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;
  NodeWeight _max_node_weight = kInvalidNodeWeight;

  StaticArray<NodeID> _permutation;
  bool _sorted;
  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;
};

namespace debug {
bool validate_graph(const Graph &graph, bool undirected = true, NodeID num_pseudo_nodes = 0);
void print_graph(const Graph &graph);
Graph sort_neighbors(Graph graph);
} // namespace debug
} // namespace kaminpar::shm
