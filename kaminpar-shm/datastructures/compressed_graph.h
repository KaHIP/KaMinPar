/*******************************************************************************
 * Compressed static graph representations.
 *
 * @file:   compressed_graph.h
 * @author: Daniel Salwasser
 * @date:   07.11.2023
 ******************************************************************************/
#pragma once

#include <vector>

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/timer.h"
#include "kaminpar-common/variable_length_codec.h"

namespace kaminpar::shm {

/*!
 * A compressed static graph that stores the nodes and edges in a compressed adjacency array. It
 * uses variable length encoding, gap encoding and interval encoding to compress the edge array.
 */
class CompressedGraph : public AbstractGraph {
public:
  using AbstractGraph::EdgeID;
  using AbstractGraph::EdgeWeight;
  using AbstractGraph::NodeID;
  using AbstractGraph::NodeWeight;

  /*!
   * Whether interval encoding is used.
   */
  static constexpr bool kIntervalEncoding = false;

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr std::size_t kIntervalLengthTreshold = 3;

  /*!
   * Constructs a new compressed graph.
   *
   * @param nodes The node array which stores for each node the offset in the compressed edges array
   * of the first edge.
   * @param compressed_edges The edge array which stores the edges for each node in a compressed
   * format.
   * @param edge_count The number of edges stored in the compressed edge array.
   * @param interval_count The number of nodes which use interval encoding.
   */
  explicit CompressedGraph(
      StaticArray<EdgeID> nodes,
      StaticArray<std::uint8_t> compressed_edges,
      StaticArray<NodeWeight> node_weights,
      StaticArray<EdgeWeight> edge_weights,
      std::size_t edge_count,
      std::size_t interval_count
  )
      : _nodes(std::move(nodes)),
        _compressed_edges(std::move(compressed_edges)),
        _node_weights(std::move(node_weights)),
        _edge_weights(std::move(edge_weights)),
        _edge_count(edge_count),
        _interval_count(interval_count) {
    KASSERT(kIntervalEncoding || interval_count == 0);

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
  };

  CompressedGraph(const CompressedGraph &) = delete;
  CompressedGraph &operator=(const CompressedGraph &) = delete;

  CompressedGraph(CompressedGraph &&) noexcept = default;
  CompressedGraph &operator=(CompressedGraph &&) noexcept = default;

  // Direct member access -- used for some "low level" operations

  [[nodiscard]] inline StaticArray<EdgeID> &raw_nodes() final {
    return _nodes;
  }

  [[nodiscard]] inline const StaticArray<EdgeID> &raw_nodes() const final {
    return _nodes;
  }

  [[nodiscard]] inline StaticArray<NodeID> &raw_edges() final {
    return _raw_edges_dummy;
  }

  [[nodiscard]] inline const StaticArray<NodeID> &raw_edges() const final {
    return _raw_edges_dummy;
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() final {
    return _node_weights;
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const final {
    return _node_weights;
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &raw_edge_weights() final {
    return _raw_edge_weights_dummy;
  }

  [[nodiscard]] inline const StaticArray<EdgeWeight> &raw_edge_weights() const final {
    return _raw_edge_weights_dummy;
  }

  [[nodiscard]] inline StaticArray<EdgeID> &&take_raw_nodes() final {
    return std::move(_nodes);
  }

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_edges() final {
    return std::move(_raw_edges_dummy);
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() final {
    return std::move(_node_weights);
  }

  [[nodiscard]] inline StaticArray<EdgeWeight> &&take_raw_edge_weights() final {
    return std::move(_raw_edge_weights_dummy);
  }

  [[nodiscard]] const StaticArray<std::uint8_t> &raw_compressed_edges() const {
    return _compressed_edges;
  }

  // Size of the graph

  [[nodiscard]] NodeID n() const final {
    return static_cast<NodeID>(_nodes.size() - 1);
  };

  [[nodiscard]] EdgeID m() const final {
    return static_cast<EdgeID>(_edge_count);
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

  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const final {
    return 0;
  }

  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const final {
    return 0;
  }

  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const final {
    return 0;
  }

  [[nodiscard]] inline NodeID degree(const NodeID node) const final {
    const std::uint8_t *data = _compressed_edges.data() + _nodes[node];

    if constexpr (kIntervalEncoding) {
      auto [degree, marker_set, len] = marked_varint_decode<NodeID>(data);
      return degree;
    } else {
      auto [degree, len] = varint_decode<NodeID>(data);
      return degree;
    }
  }

  // Parallel iteration

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  // Iterators for nodes / edges

  [[nodiscard]] IotaRange<NodeID> nodes() const final {
    return IotaRange(static_cast<NodeID>(0), n());
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<EdgeID>(0), m()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data() + _nodes[node];

    if constexpr (kIntervalEncoding) {
      auto [degree, set, degree_len] = marked_varint_decode<NodeID>(data);
      auto [first_edge, _] = varint_decode<NodeID>(data + degree_len);
      return IotaRange<EdgeID>(first_edge, first_edge + degree);
    } else {
      auto [degree, degree_len] = varint_decode<NodeID>(data);
      auto [first_edge, _] = varint_decode<NodeID>(data + degree_len);
      return IotaRange<EdgeID>(first_edge, first_edge + degree);
    }
  }

  template <typename Function>
  [[nodiscard]] inline auto transform_neighbour_range(
      const NodeID node,
      Function &&function,
      const NodeID max_neighbor_count = std::numeric_limits<NodeID>::max()
  ) const {
    const std::uint8_t *begin = _compressed_edges.data() + _nodes[node];
    const std::uint8_t *end = _compressed_edges.data() + _nodes[node + 1];

    EdgeID degree;
    bool uses_intervals = false;
    if constexpr (kIntervalEncoding) {
      auto [deg, marker_set, degree_len] = marked_varint_decode<NodeID>(begin);
      degree = deg;
      uses_intervals = marker_set;
      begin += degree_len;
    } else {
      auto [deg, degree_len] = varint_decode<NodeID>(begin);
      degree = deg;
      begin += degree_len;
    }

    auto [first_edge, first_edge_len] = varint_decode<NodeID>(begin);
    begin += first_edge_len;

    // Interval Encoding
    NodeID interval_count = 0;
    NodeID cur_interval = 0;
    NodeID cur_left_extreme = 0;
    NodeID cur_interval_len = 0;
    NodeID cur_interval_index = 0;
    NodeID previous_right_extreme = 2;

    if constexpr (kIntervalEncoding) {
      if (uses_intervals) {
        interval_count = *((NodeID *)begin);
        begin += sizeof(NodeID);
      }
    }

    // Gap Encoding
    bool is_first_gap = true;
    NodeID prev_adjacent_node = 0;

    return TransformedIotaRange2<EdgeID, std::result_of_t<Function(EdgeID, NodeID)>>(
        first_edge,
        first_edge + std::min(degree, max_neighbor_count),
        [=](const EdgeID edge) mutable {
          if constexpr (kIntervalEncoding) {
            if (uses_intervals) {
              if (cur_interval_index < cur_interval_len) {
                return function(edge, cur_left_extreme + cur_interval_index++);
              }

              if (cur_interval < interval_count) {
                auto [left_extreme_gap, left_extreme_gap_len] = varint_decode<NodeID>(begin);
                begin += left_extreme_gap_len;

                auto [interval_length_gap, interval_length_gap_len] = varint_decode<NodeID>(begin);
                begin += interval_length_gap_len;

                cur_left_extreme = left_extreme_gap + previous_right_extreme - 2;
                cur_interval_len = interval_length_gap + kIntervalLengthTreshold;
                previous_right_extreme = cur_left_extreme + cur_interval_len - 1;

                cur_interval += 1;
                cur_interval_index = 1;
                return function(edge, cur_left_extreme);
              }
            }
          }

          if (is_first_gap) {
            is_first_gap = false;

            auto [first_gap, first_gap_len] = signed_varint_decode<NodeID>(begin);
            const NodeID first_adjacent_node = first_gap + node;

            begin += first_gap_len;
            prev_adjacent_node = first_adjacent_node;
            return function(edge, first_adjacent_node);
          }

          auto [gap, gap_len] = varint_decode<NodeID>(begin);
          const NodeID adjacent_node = gap + prev_adjacent_node;

          begin += gap_len;
          prev_adjacent_node = adjacent_node;
          return function(edge, adjacent_node);
        }
    );
  }

  [[nodiscard]] auto adjacent_nodes(const NodeID node) const {
    return transform_neighbour_range(node, [](EdgeID incident_edge, NodeID adjacent_node) {
      return adjacent_node;
    });
  }

  [[nodiscard]] auto neighbors(const NodeID node) const {
    return transform_neighbour_range(node, [](EdgeID incident_edge, NodeID adjacent_node) {
      return std::make_pair(incident_edge, adjacent_node);
    });
  }

  [[nodiscard]] auto neighbors(const NodeID node, const NodeID max_neighbor_count) const {
    return transform_neighbour_range(
        node,
        [](EdgeID incident_edge, NodeID adjacent_node) {
          return std::make_pair(incident_edge, adjacent_node);
        },
        max_neighbor_count
    );
  }

  template <typename Lambda>
  inline void pfor_neighbors(const NodeID node, const NodeID max_neighbor_count, Lambda &&l) const {
    const std::uint8_t *ptr = _compressed_edges.data() + _nodes[node];

    EdgeID degree;
    bool uses_intervals = false;
    if constexpr (kIntervalEncoding) {
      auto [deg, marker_set, degree_len] = marked_varint_decode<NodeID>(ptr);
      degree = deg;
      uses_intervals = marker_set;
      ptr += degree_len;
    } else {
      auto [deg, degree_len] = varint_decode<NodeID>(ptr);
      degree = deg;
      ptr += degree_len;
    }

    auto [first_edge, first_edge_len] = varint_decode<NodeID>(ptr);
    ptr += first_edge_len;

    const NodeID max_edges = std::min(degree, max_neighbor_count);
    NodeID edge = first_edge;

    if constexpr (kIntervalEncoding) {
      if (uses_intervals) {
        const NodeID interval_count = *((NodeID *)ptr);
        ptr += sizeof(NodeID);

        NodeID previous_right_extreme = 2;
        for (NodeID i = 0; i < interval_count; i++) {
          auto [left_extreme_gap, left_extreme_gap_len] = varint_decode<NodeID>(ptr);
          ptr += left_extreme_gap_len;

          auto [interval_length_gap, interval_length_gap_len] = varint_decode<NodeID>(ptr);
          ptr += interval_length_gap_len;

          NodeID cur_left_extreme = left_extreme_gap + previous_right_extreme - 2;
          NodeID cur_interval_len = interval_length_gap + kIntervalLengthTreshold;
          previous_right_extreme = cur_left_extreme + cur_interval_len - 1;

          for (NodeID i = 0; i < cur_interval_len; i++) {
            l(edge++, cur_left_extreme + i);

            if (edge - first_edge == max_edges) {
              return;
            }
          }
        }
      }
    }

    if (edge - first_edge == max_edges) {
      return;
    }

    auto [first_gap, first_gap_len] = signed_varint_decode<NodeID>(ptr);
    ptr += first_gap_len;

    const NodeID first_adjacent_node = first_gap + node;
    NodeID prev_adjacent_node = first_adjacent_node;

    l(edge++, first_adjacent_node);

    while (edge - first_edge < max_edges) {
      auto [gap, gap_len] = varint_decode<NodeID>(ptr);
      ptr += gap_len;

      const NodeID adjacent_node = gap + prev_adjacent_node;
      prev_adjacent_node = adjacent_node;

      l(edge++, adjacent_node);
    }
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
    return false;
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

  // Compressions statistics

  /**
   * Returns the number of nodes which use interval encoding.
   *
   * @returns The number of nodes which use interval encoding.
   */
  [[nodiscard]] std::size_t interval_count() const {
    return _interval_count;
  }

  /*!
   * Returns the amount memory in bytes used by the data structure.
   *
   * @return The amount memory in bytes used by the data structure.
   */
  [[nodiscard]] std::size_t used_memory() const {
    return _nodes.size() * sizeof(EdgeID) + _compressed_edges.size() * sizeof(std::uint8_t) +
           _node_weights.size() * sizeof(NodeWeight) + _edge_weights.size() * sizeof(EdgeWeight);
  }

private:
  StaticArray<EdgeID> _nodes;
  StaticArray<std::uint8_t> _compressed_edges;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;
  NodeWeight _max_node_weight = kInvalidNodeWeight;

  StaticArray<NodeID> _permutation;

  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;

  const std::size_t _edge_count;
  const std::size_t _interval_count;

  StaticArray<NodeID> _raw_edges_dummy{};
  StaticArray<EdgeWeight> _raw_edge_weights_dummy{};

  void init_degree_buckets() {
    KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

    if (sorted()) {
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
};

/*!
 * A builder that constructs compressed graphs in a single read pass. It does this by overcommiting
 * memory for the compressed edge array.
 */
class CompressedGraphBuilder {
public:
  /*!
   * Compresses a graph in compressed sparse row format.
   *
   * @param graph The graph to compress.
   * @return The compressed input graph.
   */
  static CompressedGraph compress(const CSRGraph &graph);

  /*!
   * Initializes the builder by allocating memory for the various arrays.
   *
   * @param node_count The number of nodes of the graph to compress.
   * @param edge_count The number of edges of the graph to compress.
   * @param store_node_weights Whether node weights are stored.
   * @param store_edge_weights Whether edge weights are stored.
   */
  void init(
      std::size_t node_count,
      std::size_t edge_count,
      bool store_node_weights,
      bool store_edge_weights
  );

  /*!
   * Adds a node to the compressed graph, sorting the neighbourhood vector.
   *
   * @param node The node to add.
   * @param neighbourhood The neighbourhood of the node to add, i.e. the adjacent nodes.
   */
  void add_node(const NodeID node, std::vector<NodeID> &neighbourhood);

  /*!
   * Sets the weight of a node.
   *
   * @param node The node whose weight is to be set.
   * @param weight The weight to be set.
   */
  void set_node_weight(const NodeID node, const NodeWeight weight);

  /*!
   * Sets the weight of an edge.
   *
   * @param edge The edge whose weight is to be set.
   * @param weight The weight to be set.
   */
  void set_edge_weight(const EdgeID edge, const EdgeWeight weight);

  /*!
   * Builds the compressed graph. The builder must then be reinitialized in order to compress
   * another graph.
   *
   * @return The compressed graph that has been build.
   */
  CompressedGraph build();

  /*!
   * Returns the total weight of the nodes that have been added.
   *
   * @return The total weight of the nodes that have been added.
   */
  std::int64_t total_node_weight() const;

  /*!
   * Returns the total weight of the edges that have been added.
   *
   * @return The total weight of the edges that have been added.
   */
  std::int64_t total_edge_weight() const;

private:
  StaticArray<EdgeID> _nodes;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  std::int64_t _total_node_weight;
  std::int64_t _total_edge_weight;

  std::uint8_t *_compressed_edges;
  std::uint8_t *_cur_compressed_edges;
  std::size_t _stored_bytes;

  EdgeID _edge_count;
  std::size_t _interval_count;
};

} // namespace kaminpar::shm
