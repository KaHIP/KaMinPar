/*******************************************************************************
 * Static uncompressed CSR graph data structure.
 *
 * @file:   csr_graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <numeric>
#include <utility>
#include <vector>

#include <kassert/kassert.hpp>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/tags.h"

namespace kaminpar::shm {
class CSRGraph : public AbstractGraph {
public:
  // Data types used by this graph
  using AbstractGraph::EdgeID;
  using AbstractGraph::EdgeWeight;
  using AbstractGraph::NodeID;
  using AbstractGraph::NodeWeight;

  CSRGraph(
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  CSRGraph(
      tag::Sequential,
      StaticArray<EdgeID> nodes,
      StaticArray<NodeID> edges,
      StaticArray<NodeWeight> node_weights = {},
      StaticArray<EdgeWeight> edge_weights = {},
      bool sorted = false
  );

  CSRGraph(const CSRGraph &) = delete;
  CSRGraph &operator=(const CSRGraph &) = delete;

  CSRGraph(CSRGraph &&) noexcept = default;
  CSRGraph &operator=(CSRGraph &&) noexcept = default;

  ~CSRGraph() override = default;

  // Direct member access -- used for some "low level" operations
  [[nodiscard]] inline StaticArray<EdgeID> &raw_nodes() final {
    return _nodes;
  }

  [[nodiscard]] inline const StaticArray<EdgeID> &raw_nodes() const final {
    return _nodes;
  }

  [[nodiscard]] inline StaticArray<NodeID> &raw_edges() final {
    return _edges;
  }

  [[nodiscard]] inline const StaticArray<NodeID> &raw_edges() const final {
    return _edges;
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() final {
    return _node_weights;
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const final {
    return _node_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &raw_edge_weights() final {
    return _edge_weights;
  }

  [[nodiscard]] inline const StaticArray<EdgeWeight> &raw_edge_weights() const final {
    return _edge_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeID> &&take_raw_nodes() final {
    return std::move(_nodes);
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_edges() final {
    return std::move(_edges);
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() final {
    return std::move(_node_weights);
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &&take_raw_edge_weights() final {
    return std::move(_edge_weights);
  }

  // Size of the graph
  [[nodiscard]] inline NodeID n() const final {
    return static_cast<NodeID>(_nodes.size() - 1);
  }

  [[nodiscard]] inline EdgeID m() const final {
    return static_cast<EdgeID>(_edges.size());
  }

  // Node and edge weights
  [[nodiscard]] inline bool is_node_weighted() const final {
    return static_cast<NodeWeight>(n()) != total_node_weight();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
    return is_node_weighted() ? _node_weights[u] : 1;
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const final {
    return _max_node_weight;
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _total_node_weight;
  }

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return static_cast<EdgeWeight>(m()) != total_edge_weight();
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const final {
    return is_edge_weighted() ? _edge_weights[e] : 1;
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _total_edge_weight;
  }

  // Low-level access to the graph structure
  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return static_cast<NodeID>(_nodes[u + 1] - _nodes[u]);
  }

  // Parallel iteration
  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  // Iterators for nodes / edges
  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return {static_cast<NodeID>(0), n()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<EdgeID>(0), m()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const {
    return {_nodes[u], _nodes[u + 1]};
  }

  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const {
    return TransformedIotaRange2<EdgeID, NodeID>(_nodes[u], _nodes[u + 1], [this](const EdgeID e) {
      return _edges[e];
    });
  }

  [[nodiscard]] inline auto neighbors(const NodeID u) const {
    return TransformedIotaRange2<EdgeID, std::pair<EdgeID, NodeID>>(
        _nodes[u], _nodes[u + 1], [this](const EdgeID e) { return std::make_pair(e, _edges[e]); }
    );
  }

  [[nodiscard]] inline auto neighbors(const NodeID u, const NodeID max_neighbor_count) const {
    const EdgeID from = _nodes[u];
    const EdgeID to = from + std::min(degree(u), max_neighbor_count);

    return TransformedIotaRange2<EdgeID, std::pair<EdgeID, NodeID>>(
        from, to, [this](const EdgeID e) { return std::make_pair(e, _edges[e]); }
    );
  }

  template <typename Lambda>
  inline void pfor_neighbors(const NodeID u, const NodeID max_neighbor_count, Lambda &&l) const {
    const EdgeID from = _nodes[u];
    const EdgeID to = from + std::min(degree(u), max_neighbor_count);

    tbb::parallel_for(from, to, [&](const EdgeID e) { l(e, _edges[e]); });
  }

  // Graph permutation
  inline void set_permutation(StaticArray<NodeID> permutation) final {
    _permutation = std::move(permutation);
  }

  [[nodiscard]] inline bool permuted() const final {
    return !_permutation.empty();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const final {
    return _permutation[u];
  }

  // Degree buckets
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const final {
    return _buckets[bucket + 1] - _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const final {
    return _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const final {
    return first_node_in_bucket(bucket + 1);
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const final {
    return _number_of_buckets;
  }

  [[nodiscard]] inline bool sorted() const final {
    return _sorted;
  }

  void update_total_node_weight() final;

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
} // namespace kaminpar::shm
