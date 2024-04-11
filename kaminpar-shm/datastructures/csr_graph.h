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
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {

template <template <typename> typename Container, template <typename> typename CompactContainer>
class AbstractCSRGraph : public AbstractGraph {
public:
  // Data types used by this graph
  using AbstractGraph::EdgeID;
  using AbstractGraph::EdgeWeight;
  using AbstractGraph::NodeID;
  using AbstractGraph::NodeWeight;

  // Tag for the sequential ctor.
  struct seq {};

  AbstractCSRGraph(
      Container<EdgeID> nodes,
      CompactContainer<NodeID> edges,
      Container<NodeWeight> node_weights = {},
      CompactContainer<EdgeWeight> edge_weights = {},
      bool sorted = false
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

  AbstractCSRGraph(
      seq,
      Container<EdgeID> nodes,
      CompactContainer<NodeID> edges,
      Container<NodeWeight> node_weights = {},
      CompactContainer<EdgeWeight> edge_weights = {},
      bool sorted = false
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

  AbstractCSRGraph(const AbstractCSRGraph &) = delete;
  AbstractCSRGraph &operator=(const AbstractCSRGraph &) = delete;

  AbstractCSRGraph(AbstractCSRGraph &&) noexcept = default;
  AbstractCSRGraph &operator=(AbstractCSRGraph &&) noexcept = default;

  ~AbstractCSRGraph() override = default;

  // Direct member access -- used for some "low level" operations
  [[nodiscard]] inline Container<EdgeID> &raw_nodes() {
    return _nodes;
  }

  [[nodiscard]] inline const Container<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  [[nodiscard]] inline CompactContainer<NodeID> &raw_edges() {
    return _edges;
  }

  [[nodiscard]] inline const CompactContainer<NodeID> &raw_edges() const {
    return _edges;
  }

  [[nodiscard]] inline Container<NodeWeight> &raw_node_weights() {
    return _node_weights;
  }

  [[nodiscard]] inline const Container<NodeWeight> &raw_node_weights() const {
    return _node_weights;
  }

  [[nodiscard]] inline CompactContainer<EdgeWeight> &raw_edge_weights() {
    return _edge_weights;
  }

  [[nodiscard]] inline const CompactContainer<EdgeWeight> &raw_edge_weights() const {
    return _edge_weights;
  }

  [[nodiscard]] inline Container<EdgeID> &&take_raw_nodes() {
    return std::move(_nodes);
  }

  [[nodiscard]] inline CompactContainer<NodeID> &&take_raw_edges() {
    return std::move(_edges);
  }

  [[nodiscard]] inline Container<NodeWeight> &&take_raw_node_weights() {
    return std::move(_node_weights);
  }

  [[nodiscard]] inline CompactContainer<EdgeWeight> &&take_raw_edge_weights() {
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
  [[nodiscard]] inline bool node_weighted() const final {
    return static_cast<NodeWeight>(n()) != total_node_weight();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
    KASSERT(!node_weighted() || u < _node_weights.size());
    return node_weighted() ? _node_weights[u] : 1;
  }

  [[nodiscard]] inline NodeWeight max_node_weight() const final {
    return _max_node_weight;
  }

  [[nodiscard]] inline NodeWeight total_node_weight() const final {
    return _total_node_weight;
  }

  [[nodiscard]] inline bool edge_weighted() const final {
    return static_cast<EdgeWeight>(m()) != total_edge_weight();
  }

  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const final {
    KASSERT(!edge_weighted() || e < _edge_weights.size());
    return edge_weighted() ? _edge_weights[e] : 1;
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _total_edge_weight;
  }

  // Low-level access to the graph structure
  [[nodiscard]] inline NodeID max_degree() const final {
    return _max_degree;
  }

  [[nodiscard]] inline NodeID degree(const NodeID u) const final {
    return static_cast<NodeID>(_nodes[u + 1] - _nodes[u]);
  }

  // This function is not part of the Graph interface:
  [[nodiscard]] EdgeID first_edge(const NodeID u) const {
    return _nodes[u];
  }

  // This function is not part of the Graph interface:
  [[nodiscard]] EdgeID first_invalid_edge(const NodeID u) const {
    return _nodes[u + 1];
  }

  // This function is not part of the Graph interface:
  [[nodiscard]] NodeID edge_target(const EdgeID e) const {
    return _edges[e];
  }

  // Iterators for nodes / edges
  [[nodiscard]] inline IotaRange<NodeID> nodes() const final {
    return {static_cast<NodeID>(0), n()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<EdgeID>(0), m()};
  }

  // Parallel iteration
  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  // Graph operations
  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return {_nodes[u], _nodes[u + 1]};
  }

  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return TransformedIotaRange(_nodes[u], _nodes[u + 1], [this](const EdgeID e) {
      return _edges[e];
    });
  }

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    KASSERT(u + 1 < _nodes.size());

    const EdgeID from = _nodes[u];
    const EdgeID to = _nodes[u + 1];
    for (EdgeID edge = from; edge < to; ++edge) {
      l(_edges[edge]);
    }
  }

  [[nodiscard]] inline auto neighbors(const NodeID u) const {
    KASSERT(u + 1 < _nodes.size());
    return TransformedIotaRange(_nodes[u], _nodes[u + 1], [this](const EdgeID e) {
      return std::make_pair(e, _edges[e]);
    });
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    KASSERT(u + 1 < _nodes.size());

    const EdgeID from = _nodes[u];
    const EdgeID to = _nodes[u + 1];
    for (EdgeID edge = from; edge < to; ++edge) {
      l(edge, _edges[edge]);
    }
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_neighbor_count, Lambda &&l) const {
    KASSERT(u + 1 < _nodes.size());
    constexpr bool non_stoppable =
        std::is_void<std::invoke_result_t<Lambda, EdgeID, NodeID>>::value;

    const EdgeID from = _nodes[u];
    const EdgeID to = from + std::min(degree(u), max_neighbor_count);

    for (EdgeID edge = from; edge < to; ++edge) {
      if constexpr (non_stoppable) {
        l(edge, _edges[edge]);
      } else {
        if (l(edge, _edges[edge])) {
          return;
        }
      }
    }
  }

  template <typename Lambda>
  inline void pfor_neighbors(
      const NodeID u, const NodeID max_neighbor_count, const NodeID grainsize, Lambda &&l
  ) const {
    KASSERT(u + 1 < _nodes.size());

    const EdgeID from = _nodes[u];
    const EdgeID to = from + std::min(degree(u), max_neighbor_count);

    tbb::parallel_for(
        tbb::blocked_range<EdgeID>(from, to, grainsize),
        [&](const tbb::blocked_range<EdgeID> range) {
          const auto end = range.end();

          invoke_maybe_indirect<std::is_invocable_v<Lambda, EdgeID, NodeID>>(
              std::forward<Lambda>(l),
              [&](auto &&l2) {
                for (EdgeID e = range.begin(); e < end; ++e) {
                  l2(e, _edges[e]);
                }
              }
          );
        }
    );
  }

  // Graph permutation
  inline void set_permutation(StaticArray<NodeID> permutation) final {
    _permutation = std::move(permutation);
  }

  [[nodiscard]] inline bool permuted() const final {
    return !_permutation.empty();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID u) const final {
    KASSERT(u < _permutation.size());
    return _permutation[u];
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_permutation() final {
    return std::move(_permutation);
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

  void update_total_node_weight() final {
    if (_node_weights.empty()) {
      _total_node_weight = n();
      _max_node_weight = 1;
    } else {
      _total_node_weight =
          std::accumulate(_node_weights.begin(), _node_weights.end(), static_cast<NodeWeight>(0));
      _max_node_weight = *std::max_element(_node_weights.begin(), _node_weights.end());
    }
  }

  void remove_isolated_nodes(const NodeID isolated_nodes) {
    KASSERT(sorted());

    if (isolated_nodes == 0) {
      return;
    }

    const NodeID new_n = n() - isolated_nodes;
    _nodes.restrict(new_n + 1);
    if (!_node_weights.empty()) {
      _node_weights.restrict(new_n);
    }

    update_total_node_weight();

    // Update degree buckets
    for (std::size_t i = 0; i < _buckets.size() - 1; ++i) {
      _buckets[1 + i] -= isolated_nodes;
    }

    // If the graph has only isolated nodes then there are no buckets afterwards
    if (_number_of_buckets == 1) {
      _number_of_buckets = 0;
    }
  }

  void integrate_isolated_nodes() {
    KASSERT(sorted());

    const NodeID nonisolated_nodes = n();
    _nodes.unrestrict();
    _node_weights.unrestrict();

    const NodeID isolated_nodes = n() - nonisolated_nodes;
    update_total_node_weight();

    // Update degree buckets
    for (std::size_t i = 0; i < _buckets.size() - 1; ++i) {
      _buckets[1 + i] += isolated_nodes;
    }

    // If the graph has only isolated nodes then there is one afterwards
    if (_number_of_buckets == 0) {
      _number_of_buckets = 1;
    }
  }

  std::size_t node_id_byte_width() const {
    if constexpr (std::is_same_v<CompactContainer<NodeID>, CompactStaticArray<NodeID>>) {
      return _edges.byte_width();
    }

    return sizeof(NodeID);
  }

  std::size_t edge_weight_byte_width() const {
    if constexpr (std::is_same_v<CompactContainer<EdgeWeight>, CompactStaticArray<EdgeWeight>>) {
      return _edge_weights.byte_width();
    }

    return sizeof(EdgeWeight);
  }

private:
  void init_degree_buckets() {
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

  Container<EdgeID> _nodes;
  CompactContainer<NodeID> _edges;
  Container<NodeWeight> _node_weights;
  CompactContainer<EdgeWeight> _edge_weights;

  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;
  NodeWeight _max_node_weight = kInvalidNodeWeight;

  NodeID _max_degree;

  StaticArray<NodeID> _permutation;
  bool _sorted;
  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;
};

using CSRGraph = AbstractCSRGraph<StaticArray, StaticArray>;
using CompactCSRGraph = AbstractCSRGraph<StaticArray, CompactStaticArray>;

namespace debug {

bool validate_graph(const CSRGraph &graph, bool undirected = true, NodeID num_pseudo_nodes = 0);
CSRGraph sort_neighbors(CSRGraph graph);

} // namespace debug

} // namespace kaminpar::shm
