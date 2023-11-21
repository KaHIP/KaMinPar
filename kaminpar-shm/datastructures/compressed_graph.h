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

  using VarLengthCodec = VarIntCodec;

  /*!
   * Whether interval encoding is used.
   */
  static constexpr bool IntervalEncoding = false;

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr std::size_t kIntervalLengthTreshold = 3;

  /**
   * Compresses a graph.
   *
   * @param graph The graph to compress.
   * @return The compressed input graph.
   */
  static CompressedGraph compress(const CSRGraph &graph) {
    SCOPED_HEAP_PROFILER("Compress graph");
    SCOPED_TIMER("Compress graph");

    auto iterate = [&](auto &&handle_node,
                       auto &&handle_interval,
                       auto &&handle_first_gap,
                       auto &&handle_remaining_gap) {
      std::vector<NodeID> buffer;
      EdgeID first_edge = 0;

      for (const NodeID node : graph.nodes()) {
        const NodeID degree = graph.degree(node);
        handle_node(node, degree, first_edge);

        if (degree == 0) {
          continue;
        }

        first_edge += degree;
        for (const NodeID adjacent_node : graph.adjacent_nodes(node)) {
          buffer.push_back(adjacent_node);
        }

        // Sort the adjacent nodes in ascending order.
        std::sort(buffer.begin(), buffer.end());

        // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
        // least kIntervalLengthTreshold. Instead of storing all nodes, only store a representation
        // of the left extreme i and the length j - i + 1. Left extremes are compressed using the
        // differences between each left extreme and the previous right extreme minus 2 (because
        // there must be at least one integer between the end of an interval and the beginning of
        // the next one), except the first left extreme which is stored directly. The lengths are
        // decremented by kIntervalLengthTreshold, the minimum length of an interval.
        if constexpr (IntervalEncoding) {
          if (buffer.size() > 1) {
            NodeID previous_right_extreme = 2;
            std::size_t interval_len = 1;

            NodeID prev_adjacent_node = *buffer.begin();
            for (auto iter = buffer.begin() + 1; iter != buffer.end(); ++iter) {
              const NodeID adjacent_node = *iter;

              if (prev_adjacent_node + 1 == adjacent_node) {
                interval_len++;

                // The interval ends if there are no more nodes or the next node is not the
                // increment of the current node.
                if (iter + 1 == buffer.end() || adjacent_node + 1 != *(iter + 1)) {
                  if (interval_len >= kIntervalLengthTreshold) {
                    const NodeID left_extreme = adjacent_node + 1 - interval_len;
                    const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
                    const std::size_t interval_length_gap = interval_len - kIntervalLengthTreshold;

                    handle_interval(left_extreme_gap, interval_length_gap);

                    previous_right_extreme = adjacent_node;
                    iter = buffer.erase(iter - interval_len + 1, iter + 1);
                    if (iter == buffer.end()) {
                      break;
                    }
                  }

                  interval_len = 1;
                }
              }

              prev_adjacent_node = adjacent_node;
            }

            // If all incident edges have been compressed using intervals then gap encoding cannot
            // be applied. Thus, go to the next node.
            if (buffer.empty()) {
              continue;
            }
          }
        }

        // Store the remaining adjacent node using gap encoding. That is instead of storing the
        // nodes v_1, v_2, ..., v_{k - 1}, v_k directly, store the gaps v_1 - u, v_2 - v_1, ..., v_k
        // - v_{k - 1} between the nodes, where u is the source node. Note that all gaps except the
        // first one have to be positive as we sorted the nodes in ascending order. Thus, only for
        // the first gap the sign is additionally stored.
        const NodeID first_adjacent_node = *buffer.begin();
        // TODO: Does the value range cover everything s.t. a over- or underflow cannot happen?
        const std::make_signed_t<NodeID> first_gap = first_adjacent_node - node;
        handle_first_gap(first_gap);

        NodeID prev_adjacent_node = first_adjacent_node;
        const auto iter_end = buffer.end();
        for (auto iter = buffer.begin() + 1; iter != iter_end; ++iter) {
          const NodeID adjacent_node = *iter;
          const NodeID gap = adjacent_node - prev_adjacent_node;

          handle_remaining_gap(gap);
          prev_adjacent_node = adjacent_node;
        }

        buffer.clear();
      }
    };

    // First iterate over all nodes and their adjacent nodes. In the process calculate the number of
    // intervalls to store compressed for each node and store the number temporarily in the nodes
    // array. Additionally calculate the needed capacity for the compressed edge array.
    RECORD("nodes") StaticArray<EdgeID> nodes(graph.n() + 1);

    NodeID cur_node;
    std::size_t edge_capacity = 0;
    iterate(
        [&](auto node, auto degree, auto first_edge) {
          cur_node = node;

          if constexpr (IntervalEncoding) {
            edge_capacity += VarLengthCodec::length_marker(degree);
          } else {
            edge_capacity += VarLengthCodec::length(degree);
          }

          edge_capacity += VarLengthCodec::length(first_edge);
        },
        [&](auto left_extreme_gap, auto interval_length_gap) {
          nodes[cur_node] += 1;
          edge_capacity += VarLengthCodec::length(left_extreme_gap);
          edge_capacity += VarLengthCodec::length(interval_length_gap);
        },
        [&](auto first_gap) { edge_capacity += VarLengthCodec::length_signed(first_gap); },
        [&](auto gap) { edge_capacity += VarLengthCodec::length(gap); }
    );

    if constexpr (IntervalEncoding) {
      auto iter_end = nodes.end();
      for (auto iter = nodes.begin(); iter + 1 != iter_end; ++iter) {
        const EdgeID number_of_intervalls = *iter;

        if (number_of_intervalls > 0) {
          edge_capacity += VarLengthCodec::length(number_of_intervalls);
        }
      }
    }

    // In the second iteration fill the nodes and compressed edge array with data.
    RECORD("compressed_edges") StaticArray<std::uint8_t> compressed_edges(edge_capacity);
    std::size_t interval_count = 0;

    uint8_t *edges = compressed_edges.data();
    iterate(
        [&](auto node, auto degree, auto first_edge) {
          const EdgeID number_of_intervalls = nodes[node];
          nodes[node] = static_cast<EdgeID>(edges - compressed_edges.data());

          if constexpr (IntervalEncoding) {
            edges += VarLengthCodec::encode_with_marker(degree, number_of_intervalls > 0, edges);
          } else {
            edges += VarLengthCodec::encode(degree, edges);
          }

          edges += VarLengthCodec::encode(first_edge, edges);

          if constexpr (IntervalEncoding) {
            if (number_of_intervalls > 0) {
              edges += VarLengthCodec::encode(number_of_intervalls, edges);
              interval_count++;
            }
          }
        },
        [&](auto left_extreme_gap, auto interval_length_gap) {
          edges += VarLengthCodec::encode(left_extreme_gap, edges);
          edges += VarLengthCodec::encode(interval_length_gap, edges);
        },
        [&](auto first_gap) { edges += VarLengthCodec::encode_signed(first_gap, edges); },
        [&](auto gap) { edges += VarLengthCodec::encode(gap, edges); }
    );
    nodes[nodes.size() - 1] = compressed_edges.size();

    return CompressedGraph(
        std::move(nodes),
        std::move(compressed_edges),
        static_array::copy(graph.raw_node_weights()),
        static_array::copy(graph.raw_edge_weights()),
        graph.m(),
        interval_count
    );
  }

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
    KASSERT(IntervalEncoding || interval_count == 0);

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

    if constexpr (IntervalEncoding) {
      auto [degree, marker_set, len] = VarLengthCodec::template decode_with_marker<NodeID>(data);
      return degree;
    } else {
      auto [degree, len] = VarLengthCodec::template decode<NodeID>(data);
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

    if constexpr (IntervalEncoding) {
      auto [degree, set, degree_len] = VarLengthCodec::template decode_with_marker<NodeID>(data);
      auto [first_edge, _] = VarLengthCodec::template decode<NodeID>(data + degree_len);
      return IotaRange<EdgeID>(first_edge, first_edge + degree);
    } else {
      auto [degree, degree_len] = VarLengthCodec::template decode<NodeID>(data);
      auto [first_edge, _] = VarLengthCodec::template decode<NodeID>(data + degree_len);
      return IotaRange<EdgeID>(first_edge, first_edge + degree);
    }
  }

  template <typename Function>
  [[nodiscard]] inline auto transform_neighbour_range(const NodeID node, Function function) const {
    const std::uint8_t *begin = _compressed_edges.data() + _nodes[node];
    const std::uint8_t *end = _compressed_edges.data() + _nodes[node + 1];

    EdgeID degree;
    bool uses_intervals = false;
    if constexpr (IntervalEncoding) {
      auto [deg, marker_set, degree_len] =
          VarLengthCodec::template decode_with_marker<NodeID>(begin);
      degree = deg;
      uses_intervals = marker_set;
      begin += degree_len;
    } else {
      auto [deg, degree_len] = VarLengthCodec::template decode<NodeID>(begin);
      degree = deg;
      begin += degree_len;
    }

    auto [first_edge, first_edge_len] = VarLengthCodec::template decode<NodeID>(begin);
    begin += first_edge_len;

    // Interval Encoding
    NodeID interval_count = 0;
    NodeID cur_interval = 0;
    NodeID cur_left_extreme = 0;
    NodeID cur_interval_len = 0;
    NodeID cur_interval_index = 0;
    NodeID previous_right_extreme = 2;

    if constexpr (IntervalEncoding) {
      if (uses_intervals) {
        auto [count, count_len] = VarLengthCodec::template decode<NodeID>(begin);
        interval_count = count;
        begin += count_len;
      }
    }

    // Gap Encoding
    bool is_first_gap = true;
    NodeID prev_adjacent_node = 0;

    return TransformedIotaRange2<EdgeID, std::result_of_t<Function(EdgeID, NodeID)>>(
        first_edge,
        first_edge + degree,
        [=](const EdgeID edge) mutable {
          if constexpr (IntervalEncoding) {
            if (uses_intervals) {
              if (cur_interval_index < cur_interval_len) {
                return function(edge, cur_left_extreme + cur_interval_index++);
              }

              if (cur_interval < interval_count) {
                auto [left_extreme_gap, left_extreme_gap_len] =
                    VarLengthCodec::template decode<NodeID>(begin);
                begin += left_extreme_gap_len;

                auto [interval_length_gap, interval_length_gap_len] =
                    VarLengthCodec::template decode<NodeID>(begin);
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

            auto [first_gap, first_gap_len] = VarLengthCodec::template decode_signed<NodeID>(begin);
            const NodeID first_adjacent_node = first_gap + node;

            begin += first_gap_len;
            prev_adjacent_node = first_adjacent_node;
            return function(edge, first_adjacent_node);
          }

          auto [gap, gap_len] = VarLengthCodec::template decode<NodeID>(begin);
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

} // namespace kaminpar::shm
