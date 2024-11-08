/*******************************************************************************
 * Compressed neighborhoods of a static graph.
 *
 * @file:   compressed_neighborhoods.h
 * @author: Daniel Salwasser
 * @date:   08.07.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/streamvbyte.h"
#include "kaminpar-common/graph_compression/varint.h"
#include "kaminpar-common/graph_compression/varint_rle.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/ranges.h"

#define INVOKE_CALLBACKU(adjacent_node)                                                            \
  if constexpr (kNonStoppable) {                                                                   \
    callback(adjacent_node);                                                                       \
  } else {                                                                                         \
    const bool stop = callback(adjacent_node);                                                     \
    if (stop) [[unlikely]] {                                                                       \
      return true;                                                                                 \
    }                                                                                              \
  }

#define INVOKE_CALLBACKW(adjacent_node)                                                            \
  static_assert(kCompressEdgeWeights);                                                             \
  const SignedEdgeWeight edge_weight_gap = signed_varint_decode<SignedEdgeWeight>(&node_data);     \
  EdgeWeight edge_weight = static_cast<EdgeWeight>(edge_weight_gap + prev_edge_weight);            \
                                                                                                   \
  if constexpr (kNonStoppable) {                                                                   \
    callback(adjacent_node, edge_weight);                                                          \
  } else {                                                                                         \
    const bool stop = callback(adjacent_node, edge_weight);                                        \
    if (stop) [[unlikely]] {                                                                       \
      return true;                                                                                 \
    }                                                                                              \
  }                                                                                                \
                                                                                                   \
  prev_edge_weight = edge_weight;

#define INVOKE_CALLBACK(adjacent_node)                                                             \
  if constexpr (kHasEdgeWeights) {                                                                 \
    INVOKE_CALLBACKW(adjacent_node);                                                               \
  } else {                                                                                         \
    INVOKE_CALLBACKU(adjacent_node);                                                               \
  }

namespace kaminpar {

/*!
 * The neighborhoods of a graph, which are stored in compressed format through variable-length
 * encoding, gap encoding, interval encoding and high-degree encoding.
 *
 * @tparam NodeID The type of integer to use to identify a node.
 * @tparam EdgeID The type of integer to use to identify an edge.
 * @tparam EdgeWeight The type of integer to use for edge weights.
 */
template <typename NodeID, typename EdgeID, typename EdgeWeight> class CompressedNeighborhoods {
  static_assert(std::numeric_limits<NodeID>::is_integer);
  static_assert(std::numeric_limits<EdgeID>::is_integer);
  static_assert(std::numeric_limits<EdgeWeight>::is_integer);

  using SignedNodeID = std::int64_t;
  using SignedEdgeWeight = std::make_signed_t<EdgeWeight>;

  using StreamVByteGapDecoder =
      streamvbyte::StreamVByteDecoder<NodeID, false, streamvbyte::DifferentialCodingKind::D1>;

  using StreamVByteGapAndWeightsDecoder =
      streamvbyte::StreamVByteDecoder<NodeID, true, streamvbyte::DifferentialCodingKind::D2>;

  static constexpr EdgeWeight kDefaultEdgeWeight = 1;

public:
  /*!
   * Whether edge weights are compressed.
   */
#ifdef KAMINPAR_COMPRESSION_EDGE_WEIGHTS
  static constexpr bool kCompressEdgeWeights = true;
#else
  static constexpr bool kCompressEdgeWeights = false;
#endif

  /*!
   * Whether high-degree encoding is used.
   */
#ifdef KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING
  static constexpr bool kHighDegreeEncoding = true;
#else
  static constexpr bool kHighDegreeEncoding = false;
#endif

  /*!
   * The minimum degree of a node to be considered high degree.
   */
  static constexpr NodeID kHighDegreeThreshold = 10000;

  /*!
   * The length of each part when splitting the neighbourhood of a high degree
   * node.
   */
  static constexpr NodeID kHighDegreePartLength = 1000;

  /*!
   * Whether interval encoding is used.
   */
#ifdef KAMINPAR_COMPRESSION_INTERVAL_ENCODING
  static constexpr bool kIntervalEncoding = true;
#else
  static constexpr bool kIntervalEncoding = false;
#endif

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr NodeID kIntervalLengthTreshold = 3;

  /*!
   * Whether run-length encoding is used.
   */
#ifdef KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING
  static constexpr bool kRunLengthEncoding = true;
#else
  static constexpr bool kRunLengthEncoding = false;
#endif

  /*!
   * Whether StreamVByte encoding is used.
   */
#ifdef KAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING
  static constexpr bool kStreamVByteEncoding = true;
#else
  static constexpr bool kStreamVByteEncoding = false;
#endif

  /*!
   * The minimum number of adjacent nodes required to use StreamVByte encoding.
   */
  static constexpr NodeID kStreamVByteThreshold = 3;

  static_assert(
      !kRunLengthEncoding || !kStreamVByteEncoding,
      "Either run-length or StreamVByte encoding can be used for varints "
      "but not both."
  );

  static_assert(
      !kRunLengthEncoding || !kCompressEdgeWeights,
      "Run-length cannot be used together with compressed edge weights."
  );

  static_assert(
      !kStreamVByteEncoding || !kCompressEdgeWeights || sizeof(NodeID) == sizeof(EdgeWeight),
      "StreamVByte together with compressed edge weights can only be used when the node IDs and "
      "edge weights have the same width."
  );

  /*!
   * Constructs a new CompressedNeighborhoods.
   *
   * @param nodes The offsets for each node into the compressed edges where the corresponding
   * adjacent nodes and edge weights are encoded.
   * @param compressed_edges The edges and edge weights in compresed format.
   * @param edge_weights The edge weights of the graph, which is only used when the graph has edge
   * weights and edg weight compression is disabled.
   * @param max_degree The maximum degree of the nodes.
   * @param num_edges The number of edges.
   * @param has_edge_weights Whether edge weights are stored.
   * @param total_edge_weight The total edge weight.
   * @param num_high_degree_nodes The number of nodes that have high degree.
   * @param num_high_degree_parts The total number of parts that result from splitting high degree
   * neighborhoods.
   * @param num_interval_nodes The number of nodes that have at least one interval.
   * @param num_intervals The total number of intervals.
   */
  CompressedNeighborhoods(
      CompactStaticArray<EdgeID> nodes,
      StaticArray<std::uint8_t> compressed_edges,
      StaticArray<EdgeWeight> edge_weights,
      const NodeID max_degree,
      const EdgeID num_edges,
      const bool has_edge_weights,
      const EdgeWeight total_edge_weight,
      std::size_t num_high_degree_nodes,
      std::size_t num_high_degree_parts,
      std::size_t num_interval_nodes,
      std::size_t num_intervals
  )
      : _nodes(std::move(nodes)),
        _compressed_edges(std::move(compressed_edges)),
        _edge_weights(std::move(edge_weights)),
        _num_edges(num_edges),
        _max_degree(max_degree),
        _has_edge_weights(has_edge_weights),
        _total_edge_weight(total_edge_weight),
        _num_high_degree_nodes(num_high_degree_nodes),
        _num_high_degree_parts(num_high_degree_parts),
        _num_interval_nodes(num_interval_nodes),
        _num_intervals(num_intervals) {
    KASSERT(kHighDegreeEncoding || _num_high_degree_nodes == 0);
    KASSERT(kHighDegreeEncoding || _num_high_degree_parts == 0);
    KASSERT(kIntervalEncoding || _num_interval_nodes == 0);
    KASSERT(kIntervalEncoding || _num_intervals == 0);
    KASSERT(!has_edge_weights || kCompressEdgeWeights || _edge_weights.size() == num_edges);
    KASSERT(has_edge_weights || _edge_weights.empty());
  }

  CompressedNeighborhoods(const CompressedNeighborhoods &) = delete;
  CompressedNeighborhoods &operator=(const CompressedNeighborhoods &) = delete;

  CompressedNeighborhoods(CompressedNeighborhoods &&) noexcept = default;
  CompressedNeighborhoods &operator=(CompressedNeighborhoods &&) noexcept = default;

  /*!
   * Returns the number of nodes.
   *
   * @return The number of nodes.
   */
  [[nodiscard]] EdgeID num_nodes() const {
    return _nodes.size() - 1;
  }

  /*!
   * Returns the number of edges.
   *
   * @return The number of edges.
   */
  [[nodiscard]] EdgeID num_edges() const {
    return _num_edges;
  }

  /*!
   * Returns whether the edges are weighted.
   *
   * @return Whether the edges are weighted.
   */
  [[nodiscard]] bool has_edge_weights() const {
    return _has_edge_weights;
  }

  /*!
   * Returns the total edge weight.
   *
   * @return The total edge weight.
   */
  [[nodiscard]] EdgeWeight total_edge_weight() const {
    return _total_edge_weight;
  }

  /*!
   * Returns the maximum degree of the nodes.
   *
   * @return The maximum degree of the nodes.
   */
  [[nodiscard]] NodeID max_degree() const {
    return _max_degree;
  }

  /*!
   * Returns the degree of a node.
   *
   * @param node The node whose degree is to be returned.
   * @return The degree of the node.
   */
  [[nodiscard]] NodeID degree(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data();
    const std::uint8_t *node_data = data + _nodes[node];

    if constexpr (kIntervalEncoding) {
      const auto header = marked_varint_decode<NodeID>(&node_data);
      return header.first;
    } else {
      return varint_decode<NodeID>(&node_data);
    }
  }

  /*!
   * Decodes the adjacent nodes of a node.
   *
   * @tparam Callback The type of callback to invoke with the adjacent nodes.
   * @param node The node whose adjacent nodes are to be decoded.
   * @param callback The function to invoke with each adjacent node.
   */
  template <typename Callback> void adjacent_nodes(const NodeID node, Callback &&callback) const {
    decode_adjacent_nodes<false>(node, std::forward<Callback>(callback));
  }

  /*!
   * Decodes a part of the adjacent nodes of a node.
   *
   * @tparam Callback The type of callback to invoke with the neighbor.
   * @param node The node whose neighbors are to be decoded.
   * @param max_num_neighbors The maximum number of adjacent node to be decoded.
   * @param callback The function to invoke with each neighbor.
   */
  template <typename Callback>
  void
  adjacent_nodes(const NodeID node, const NodeID max_num_neighbors, Callback &&callback) const {
    static_assert(std::is_invocable_v<Callback, NodeID, EdgeWeight>);
    constexpr bool kNonStoppable =
        std::is_void_v<std::invoke_result_t<Callback, NodeID, EdgeWeight>>;

    NodeID num_neighbors_visited = 1;
    const auto invoke_and_check = [&](const NodeID v, const EdgeWeight w) {
      bool abort = num_neighbors_visited++ >= max_num_neighbors;

      if constexpr (kNonStoppable) {
        callback(v, w);
      } else {
        abort |= callback(v, w);
      }

      return abort;
    };

    constexpr bool kInvokeDirectly = true;
    constexpr bool kParallel = false;
    if (_has_edge_weights) [[unlikely]] {
      decode<true, kInvokeDirectly, kParallel>(node, [&](const NodeID v, const EdgeWeight w) {
        return invoke_and_check(v, w);
      });
    } else {
      decode<false, kInvokeDirectly, kParallel>(node, [&](const NodeID v) {
        return invoke_and_check(v, kDefaultEdgeWeight);
      });
    }
  }

  /*!
   * Decodes the adjacent nodes of a node in parallel.
   *
   * @tparam Callback The type of callback to invoke with the adjacent nodes.
   * @param node The node whose adjacent nodes are to be decoded.
   * @param callback The function to invoke with each adjacent node.
   */
  template <typename Callback>
  void parallel_adjacent_nodes(const NodeID node, Callback &&callback) const {
    decode_adjacent_nodes<true>(node, std::forward<Callback>(callback));
  }

  /*!
   * Restricts the node array to a specific number of nodes.
   *
   * @param new_num_nodes The new number of nodes.
   */
  void restrict_nodes(const NodeID new_num_nodes) {
    _nodes.restrict(new_num_nodes);
  }

  /*!
   * Unrestricts the node array.
   */
  void unrestrict_nodes() {
    _nodes.unrestrict();
  }

  /*!
   * Returns the used memory space in bytes.
   *
   * @return The used memory space in bytes.
   */
  [[nodiscard]] std::size_t memory_space() const {
    return _nodes.memory_space() + _compressed_edges.size() +
           _edge_weights.size() * sizeof(EdgeWeight);
  }

  /*!
   * Returns the number of nodes that have high degree.
   *
   * @returns The number of nodes that have high degree.
   */
  [[nodiscard]] std::size_t num_high_degree_nodes() const {
    return _num_high_degree_nodes;
  }

  /*!
   * Returns the total number of parts that result from splitting high degree neighborhoods.
   *
   * @returns The total number of parts that result from splitting high degree neighborhoods.
   */
  [[nodiscard]] std::size_t num_high_degree_parts() const {
    return _num_high_degree_parts;
  }

  /*!
   * Returns the number of nodes that have at least one interval.
   *
   * @returns The number of nodes that have at least one interval.
   */
  [[nodiscard]] std::size_t num_interval_nodes() const {
    return _num_interval_nodes;
  }

  /*!
   * Returns the total number of intervals.
   *
   * @returns The total number of intervals.
   */
  [[nodiscard]] std::size_t num_intervals() const {
    return _num_intervals;
  }

  /*!
   * Returns ownership of the raw node array.
   *
   * @return Ownership of the raw node array.
   */
  [[nodiscard]] CompactStaticArray<EdgeID> &&take_raw_nodes() {
    return std::move(_nodes);
  }

  /*!
   * Returns a reference to the raw node array.
   *
   * @return A reference to the raw node array.
   */
  [[nodiscard]] CompactStaticArray<EdgeID> &raw_nodes() {
    return _nodes;
  }

  /*!
   * Returns a reference to the raw node array.
   *
   * @return A reference to the raw node array.
   */
  [[nodiscard]] const CompactStaticArray<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  /*!
   * Returns a reference to the raw compressed edges.
   *
   * @return A reference to the raw compressed edges.
   */
  [[nodiscard]] const StaticArray<std::uint8_t> &raw_compressed_edges() const {
    return _compressed_edges;
  }

  /*!
   * Returns a reference to the raw edge weights.
   *
   * Note that the weights are only valid when edge weight compression is enabled and when the
   * graph has edge weights.
   *
   * @return A reference to the raw edge weights.
   */
  [[nodiscard]] const StaticArray<EdgeWeight> &raw_edge_weights() const {
    return _edge_weights;
  }

private:
  template <bool kParallel, typename Callback>
  void decode_adjacent_nodes(const NodeID node, Callback &&callback) const {
    constexpr bool kInvokeDirectly = std::is_invocable_v<Callback, NodeID, EdgeWeight>;

    if (_has_edge_weights) [[unlikely]] {
      decode<true, kInvokeDirectly, kParallel>(node, std::forward<Callback>(callback));
    } else {
      if constexpr (kInvokeDirectly) {
        decode<false, kInvokeDirectly, kParallel>(node, [&](const NodeID v) {
          return callback(v, kDefaultEdgeWeight);
        });
      } else {
        decode<false, kInvokeDirectly, kParallel>(node, [&](auto &&local_decode) {
          callback([&](auto &&actual_callback) {
            local_decode([&](const NodeID v) { return actual_callback(v, kDefaultEdgeWeight); });
          });
        });
      }
    }
  }

  template <bool kHasEdgeWeights, bool kInvokeDirectly, bool kParallel, typename Callback>
  void decode(const NodeID node, Callback &&callback) const {
    const std::uint8_t *data = _compressed_edges.data();
    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];
    if (node_data == next_node_data) [[unlikely]] {
      return;
    }

    NodeID degree;
    bool has_intervals;
    if constexpr (kIntervalEncoding) {
      const auto header = marked_varint_decode<NodeID>(&node_data);
      degree = header.first;
      has_intervals = header.second;
    } else {
      degree = varint_decode<NodeID>(&node_data);
    }

    if constexpr (kHighDegreeEncoding) {
      const bool split_neighbourhood = degree >= kHighDegreeThreshold;

      if (split_neighbourhood) [[unlikely]] {
        decode_parts<kHasEdgeWeights, kInvokeDirectly, kParallel>(
            node_data, node, degree, std::forward<Callback>(callback)
        );
        return;
      }
    }

    invoke_indirect<kInvokeDirectly>(std::forward<Callback>(callback), [&](auto &&actual_callback) {
      decode_edges<kHasEdgeWeights>(
          node_data,
          node,
          degree,
          has_intervals,
          std::forward<decltype(actual_callback)>(actual_callback)
      );
    });
  }

  template <bool kHasEdgeWeights, bool kInvokeDirectly, bool kParallel, typename Callback>
  void decode_parts(
      const std::uint8_t *node_data, const NodeID node, const NodeID degree, Callback &&callback
  ) const {
    const NodeID num_parts = math::div_ceil(degree, kHighDegreePartLength);
    const auto decode_part = [&](const NodeID part) {
      NodeID part_offset = *(reinterpret_cast<const NodeID *>(node_data) + part);
      bool has_intervals;
      if constexpr (kIntervalEncoding) {
        has_intervals = math::is_msb_set(part_offset);
        part_offset &= ~math::kSetMSB<NodeID>;
      }

      const bool last_part = (part + 1) == num_parts;
      const NodeID part_degree =
          last_part ? (degree - part * kHighDegreePartLength) : kHighDegreePartLength;

      const std::uint8_t *part_data = node_data + part_offset;
      return invoke_indirect2<kInvokeDirectly, bool>(
          std::forward<Callback>(callback),
          [&](auto &&actual_callback) {
            return decode_edges<kHasEdgeWeights>(
                part_data,
                node,
                part_degree,
                has_intervals,
                std::forward<decltype(actual_callback)>(actual_callback)
            );
          }
      );
    };

    if constexpr (kParallel) {
      tbb::parallel_for<NodeID>(0, num_parts, decode_part);
    } else {
      for (NodeID part = 0; part < num_parts; ++part) {
        const bool stop = decode_part(part);
        if (stop) [[unlikely]] {
          return;
        }
      }
    }
  }

  template <bool kHasEdgeWeights, typename Callback>
  bool decode_edges(
      const std::uint8_t *node_data,
      const NodeID node,
      const NodeID degree,
      const bool has_intervals,
      Callback &&callback
  ) const {
    using CallbackReturnType = std::conditional_t<
        kHasEdgeWeights,
        std::invoke_result<Callback, NodeID, EdgeWeight>,
        std::invoke_result<Callback, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<CallbackReturnType>;

    NodeID remaining_edges = degree;
    EdgeWeight prev_edge_weight = 0;
    if constexpr (kIntervalEncoding) {
      if (has_intervals) {
        NodeID num_intervals = varint_decode<NodeID>(&node_data) + 1;
        NodeID prev_right_extreme = 0;

        do {
          const NodeID left_extreme_gap = varint_decode<NodeID>(&node_data);
          const NodeID length_gap = varint_decode<NodeID>(&node_data);

          const NodeID left_extreme = left_extreme_gap + prev_right_extreme;
          const NodeID length = length_gap + kIntervalLengthTreshold;
          prev_right_extreme = left_extreme + (length - 1) + 2;

          static_assert(kIntervalLengthTreshold == 3, "Optimized for length threshold = 3.");
          INVOKE_CALLBACK(left_extreme);
          INVOKE_CALLBACK(left_extreme + 1);
          INVOKE_CALLBACK(left_extreme + 2);

          for (NodeID j = kIntervalLengthTreshold; j < length; ++j) {
            const NodeID adjacent_node = left_extreme + j;
            INVOKE_CALLBACK(adjacent_node);
          }

          num_intervals -= 1;
          remaining_edges -= length;
        } while (num_intervals > 0);

        if (remaining_edges == 0) [[unlikely]] {
          return false;
        }
      }
    }

    const SignedNodeID first_gap = signed_varint_decode<SignedNodeID>(&node_data);
    const NodeID first_adjacent_node = static_cast<NodeID>(first_gap + node);
    INVOKE_CALLBACK(first_adjacent_node);
    remaining_edges -= 1;

    /*
    if constexpr (kRunLengthEncoding) {
      VarIntRunLengthDecoder<NodeID> rl_decoder(remaining_edges, node_data);

      bool stop = false;
      NodeID prev_adjacent_node = first_adjacent_node;
      rl_decoder.decode([&](const NodeID gap) {
        const NodeID adjacent_node = gap + prev_adjacent_node + 1;
        prev_adjacent_node = adjacent_node;

        if constexpr (kHasEdgeWeights) {
          EdgeWeight edge_weight = _edge_weights[edge];

          if constexpr (kNonStoppable) {
            callback(edge++, adjacent_node, edge_weight);
          } else {
            stop = callback(edge++, adjacent_node, edge_weight);
            return stop;
          }
        } else {
          if constexpr (kNonStoppable) {
            callback(edge++, adjacent_node);
          } else {
            stop = callback(edge++, adjacent_node);
            return stop;
          }
        }
      });

      return stop;
    } else if constexpr (kStreamVByteEncoding) {
      const NodeID num_remaining_gaps = static_cast<NodeID>(last_edge - edge);

      if (num_remaining_gaps >= kStreamVByteThreshold) {
        bool stop = false;

        if constexpr (kHasEdgeWeights) {
          StreamVByteGapAndWeightsDecoder decoder(num_remaining_gaps * 2, node_data);
          decoder.decode([&](const NodeID adjacent_node, const EdgeWeight edge_weight) {
            if constexpr (kNonStoppable) {
              callback(edge++, adjacent_node, edge_weight);
            } else {
              stop = callback(edge++, adjacent_node, edge_weight);
              return stop;
            }
          });
        } else {
          StreamVByteGapDecoder decoder(num_remaining_gaps, node_data);
          decoder.decode([&](const NodeID adjacent_node) {
            if constexpr (kNonStoppable) {
              callback(edge++, adjacent_node);
            } else {
              stop = callback(edge++, adjacent_node);
              return stop;
            }
          });
        }

        return stop;
      }
    }
    */

    NodeID prev_adjacent_node = first_adjacent_node;
    while (remaining_edges > 0) {
      const NodeID gap = varint_decode<NodeID>(&node_data);
      const NodeID adjacent_node = gap + prev_adjacent_node + 1;

      INVOKE_CALLBACK(adjacent_node);
      prev_adjacent_node = adjacent_node;
      remaining_edges -= 1;
    }

    return false;
  }

private:
  CompactStaticArray<EdgeID> _nodes;
  StaticArray<std::uint8_t> _compressed_edges;
  StaticArray<EdgeWeight> _edge_weights;

  EdgeID _num_edges;
  NodeID _max_degree;

  bool _has_edge_weights;
  EdgeWeight _total_edge_weight;

  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;
};

} // namespace kaminpar
