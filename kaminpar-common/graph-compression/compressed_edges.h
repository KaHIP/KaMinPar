#pragma once

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/varint_codec.h"
#include "kaminpar-common/varint_run_length_codec.h"
#include "kaminpar-common/varint_stream_codec.h"

namespace kaminpar {

template <typename NodeID, typename EdgeID> class CompressedEdges {
  static_assert(std::numeric_limits<NodeID>::is_integer);
  static_assert(std::numeric_limits<EdgeID>::is_integer);

public:
  using SignedID = std::int64_t;

#ifdef KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING
  /*!
   * Whether high degree encoding is used.
   */
  static constexpr bool kHighDegreeEncoding = true;
#else
  /*!
   * Whether high degree encoding is used.
   */
  static constexpr bool kHighDegreeEncoding = false;
#endif

  /*!
   * The minimum degree of a node to be considered high degree.
   */
  static constexpr NodeID kHighDegreeThreshold = 10000;

  /*!
   * The length of a part when splitting the neighbourhood of a high degree
   * node.
   */
  static constexpr NodeID kHighDegreePartLength = 1000;

#ifdef KAMINPAR_COMPRESSION_INTERVAL_ENCODING
  /*!
   * Whether interval encoding is used.
   */
  static constexpr bool kIntervalEncoding = true;
#else
  /*!
   * Whether interval encoding is used.
   */
  static constexpr bool kIntervalEncoding = false;
#endif

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr NodeID kIntervalLengthTreshold = 3;

#ifdef KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING
  /*!
   * Whether run-length encoding is used.
   */
  static constexpr bool kRunLengthEncoding = true;
#else
  /*!
   * Whether run-length encoding is used.
   */
  static constexpr bool kRunLengthEncoding = false;
#endif

#ifdef KAMINPAR_COMPRESSION_STREAM_ENCODING
  /*!
   * Whether stream encoding is used.
   */
  static constexpr bool kStreamEncoding = true;
#else
  /*!
   * Whether stream encoding is used.
   */
  static constexpr bool kStreamEncoding = false;
#endif

  static_assert(
      !kRunLengthEncoding || !kStreamEncoding,
      "Either run-length or stream encoding can be used for varints "
      "but not both."
  );

#ifdef KAMINPAR_COMPRESSION_ISOLATED_NODES_SEPARATION
  /*!
   * Whether the isolated nodes of the compressed graph are continuously stored
   * at the end of the nodes array.
   */
  static constexpr bool kIsolatedNodesSeparation = true;
#else
  /*!
   * Whether the isolated nodes of the compressed graph are continuously stored
   * at the end of the nodes array.
   */
  static constexpr bool kIsolatedNodesSeparation = false;
#endif

  CompressedEdges(const EdgeID num_edges, StaticArray<std::uint8_t> compressed_edges)
      : _num_edges(num_edges),
        _compressed_edges(std::move(compressed_edges)) {}

  CompressedEdges(const CompressedEdges &) = delete;
  CompressedEdges &operator=(const CompressedEdges &) = delete;

  CompressedEdges(CompressedEdges &&) noexcept = default;
  CompressedEdges &operator=(CompressedEdges &&) noexcept = default;

  [[nodiscard]] EdgeID num_edges() const {
    return _num_edges;
  }

  [[nodiscard]] NodeID
  degree(const NodeID node, const EdgeID edge_offset, const EdgeID next_edge_offset) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + edge_offset;
    const std::uint8_t *next_node_data = data + next_edge_offset;

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return 0;
    }

    const auto header = decode_header(node, node_data, next_node_data);
    return std::get<1>(header);
  }

  [[nodiscard]] IotaRange<EdgeID>
  incident_edges(const NodeID node, const EdgeID edge_offset, const EdgeID next_edge_offset) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + edge_offset;
    const std::uint8_t *next_node_data = data + next_edge_offset;

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return {0, 0};
    }

    const auto [first_edge, degree, _, __] = decode_header(node, node_data, next_node_data);
    return {first_edge, first_edge + degree};
  }

  template <bool kParallelDecoding = false, typename Lambda>
  void decode_neighborhood(
      const NodeID node, const EdgeID edge_offset, const EdgeID next_edge_offset, Lambda &&l
  ) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + edge_offset;
    const std::uint8_t *next_node_data = data + next_edge_offset;

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return;
    }

    const auto header = decode_header(node, node_data, next_node_data);
    const auto &edge = std::get<0>(header);
    const auto &degree = std::get<1>(header);
    const auto &uses_intervals = std::get<2>(header);
    const auto &len = std::get<3>(header);

    node_data += len;

    if constexpr (kHighDegreeEncoding) {
      if (degree >= kHighDegreeThreshold) {
        decode_parts<kParallelDecoding>(node_data, node, edge, degree, std::forward<Lambda>(l));
        return;
      }
    }

    invoke_indirect<std::is_invocable_v<Lambda, EdgeID, NodeID>>(
        std::forward<Lambda>(l),
        [&](auto &&l2) {
          decode_edges(
              node_data, node, edge, degree, uses_intervals, std::forward<decltype(l2)>(l2)
          );
        }
    );
  }

private:
  EdgeID _num_edges;
  StaticArray<std::uint8_t> _compressed_edges;

private:
  inline std::tuple<EdgeID, NodeID, bool, std::size_t> decode_header(
      const NodeID node, const std::uint8_t *node_data, const std::uint8_t *next_node_data
  ) const {
    const auto [first_edge, next_first_edge, uses_intervals, len] = [&] {
      if constexpr (kIntervalEncoding) {
        auto [first_edge, uses_intervals, len] = marked_varint_decode<EdgeID>(node_data);
        auto [next_first_edge, _, __] = marked_varint_decode<EdgeID>(next_node_data);

        return std::make_tuple(first_edge, next_first_edge, uses_intervals, len);
      } else {
        auto [first_edge, len] = varint_decode<EdgeID>(node_data);
        auto [next_first_edge, _] = varint_decode<EdgeID>(next_node_data);

        return std::make_tuple(first_edge, next_first_edge, false, len);
      }
    }();

    if constexpr (kIsolatedNodesSeparation) {
      const EdgeID ungapped_first_edge = first_edge + node;
      const NodeID degree = static_cast<NodeID>(1 + next_first_edge - first_edge);
      return std::make_tuple(ungapped_first_edge, degree, uses_intervals, len);
    } else {
      const NodeID degree = static_cast<NodeID>(next_first_edge - first_edge);
      return std::make_tuple(first_edge, degree, uses_intervals, len);
    }
  }

  template <bool parallel, typename Lambda>
  void decode_parts(
      const std::uint8_t *data,
      const NodeID node,
      const EdgeID edge,
      const NodeID degree,
      Lambda &&l
  ) const {
    const NodeID part_count = math::div_ceil(degree, kHighDegreePartLength);

    const auto iterate_part = [&](const NodeID part) {
      const NodeID part_offset = *((NodeID *)(data + sizeof(NodeID) * part));
      const std::uint8_t *part_data = data + part_offset;

      const NodeID part_count_m1 = part_count - 1;
      const bool last_part = part == part_count_m1;

      const EdgeID part_edge = edge + kHighDegreePartLength * part;
      const NodeID part_degree =
          last_part ? (degree - kHighDegreePartLength * part_count_m1) : kHighDegreePartLength;

      return invoke_indirect2<std::is_invocable_v<Lambda, EdgeID, NodeID>, bool>(
          std::forward<Lambda>(l),
          [&](auto &&l2) {
            return decode_edges(
                part_data, node, part_edge, part_degree, true, std::forward<decltype(l2)>(l2)
            );
          }
      );
    };

    if constexpr (parallel) {
      tbb::parallel_for<NodeID>(0, part_count, std::forward<decltype(iterate_part)>(iterate_part));
    } else {
      for (NodeID part = 0; part < part_count; ++part) {
        const bool stop = iterate_part(part);
        if (stop) {
          return;
        }
      }
    }
  }

  template <typename Lambda>
  bool decode_edges(
      const std::uint8_t *data,
      const NodeID node,
      EdgeID edge,
      const NodeID degree,
      bool uses_intervals,
      Lambda &&l
  ) const {
    const EdgeID max_edge = edge + degree;

    if constexpr (kIntervalEncoding) {
      if (uses_intervals) {
        const bool stop = decode_intervals(data, edge, std::forward<Lambda>(l));
        if (stop) {
          return true;
        }

        if (edge == max_edge) {
          return false;
        }
      }
    }

    return decode_gaps(data, node, edge, max_edge, std::forward<Lambda>(l));
  }

  template <typename Lambda>
  bool decode_intervals(const std::uint8_t *&data, EdgeID &edge, Lambda &&l) const {
    constexpr bool non_stoppable = std::is_void_v<std::invoke_result_t<Lambda, EdgeID, NodeID>>;

    const NodeID interval_count = *((NodeID *)data);
    data += sizeof(NodeID);

    NodeID previous_right_extreme = 2;
    for (NodeID i = 0; i < interval_count; ++i) {
      const auto [left_extreme_gap, left_extreme_gap_len] = varint_decode<NodeID>(data);
      data += left_extreme_gap_len;

      const auto [interval_length_gap, interval_length_gap_len] = varint_decode<NodeID>(data);
      data += interval_length_gap_len;

      const NodeID cur_left_extreme = left_extreme_gap + previous_right_extreme - 2;
      const NodeID cur_interval_len = interval_length_gap + kIntervalLengthTreshold;
      previous_right_extreme = cur_left_extreme + cur_interval_len - 1;

      for (NodeID j = 0; j < cur_interval_len; ++j) {
        if constexpr (non_stoppable) {
          l(edge, cur_left_extreme + j);
        } else {
          const bool stop = l(edge, cur_left_extreme + j);
          if (stop) {
            return true;
          }
        }

        edge += 1;
      }
    }

    return false;
  }

  template <typename Lambda>
  bool decode_gaps(
      const std::uint8_t *data, NodeID node, EdgeID &edge, const EdgeID max_edge, Lambda &&l
  ) const {
    constexpr bool non_stoppable = std::is_void_v<std::invoke_result_t<Lambda, EdgeID, NodeID>>;

    const auto [first_gap, first_gap_len] = signed_varint_decode<SignedID>(data);
    data += first_gap_len;

    const NodeID first_adjacent_node = static_cast<NodeID>(first_gap + node);
    NodeID prev_adjacent_node = first_adjacent_node;

    if constexpr (non_stoppable) {
      l(edge, first_adjacent_node);
    } else {
      const bool stop = l(edge, first_adjacent_node);
      if (stop) {
        return true;
      }
    }
    edge += 1;

    const auto handle_gap = [&](const NodeID gap) {
      const NodeID adjacent_node = gap + prev_adjacent_node + 1;
      prev_adjacent_node = adjacent_node;

      if constexpr (non_stoppable) {
        l(edge++, adjacent_node);
      } else {
        return l(edge++, adjacent_node);
      }
    };

    if constexpr (kRunLengthEncoding) {
      VarIntRunLengthDecoder<NodeID> rl_decoder(data, max_edge - edge);
      rl_decoder.decode(std::forward<decltype(handle_gap)>(handle_gap));
    } else if constexpr (kStreamEncoding) {
      VarIntStreamDecoder<NodeID> sv_encoder(data, max_edge - edge);
      sv_encoder.decode(std::forward<decltype(handle_gap)>(handle_gap));
    } else {
      while (edge != max_edge) {
        const auto [gap, gap_len] = varint_decode<NodeID>(data);
        data += gap_len;

        const NodeID adjacent_node = gap + prev_adjacent_node + 1;
        prev_adjacent_node = adjacent_node;

        if constexpr (non_stoppable) {
          l(edge, adjacent_node);
        } else {
          const bool stop = l(edge, adjacent_node);
          if (stop) {
            return true;
          }
        }

        edge += 1;
      }
    }

    return false;
  }
};

} // namespace kaminpar
