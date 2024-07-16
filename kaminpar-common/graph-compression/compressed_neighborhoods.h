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
#include "kaminpar-common/math.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/varint_codec.h"
#include "kaminpar-common/varint_run_length_codec.h"
#include "kaminpar-common/varint_stream_codec.h"

namespace kaminpar {

template <typename NodeID, typename EdgeID, typename EdgeWeight> class CompressedNeighborhoods {
  static_assert(std::numeric_limits<NodeID>::is_integer);
  static_assert(std::numeric_limits<EdgeID>::is_integer);
  static_assert(std::numeric_limits<EdgeWeight>::is_integer);

  struct NeighborhoodHeader {
    EdgeID first_edge;
    NodeID degree;
    bool uses_intervals;
    std::size_t length;
  };

public:
  using SignedID = std::int64_t;

  /*!
   * Whether edge weights are compressed.
   */
#ifdef KAMINPAR_COMPRESSION_EDGE_WEIGHTS
  static constexpr bool kCompressEdgeWeights = true;
#else
  static constexpr bool kCompressEdgeWeights = false;
#endif

  /*!
   * Whether high degree encoding is used.
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
   * The length of a part when splitting the neighbourhood of a high degree
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
   * Whether stream encoding is used.
   */
#ifdef KAMINPAR_COMPRESSION_STREAM_ENCODING
  static constexpr bool kStreamEncoding = true;
#else
  static constexpr bool kStreamEncoding = false;
#endif

  static_assert(
      !kRunLengthEncoding || !kStreamEncoding,
      "Either run-length or stream encoding can be used for varints "
      "but not both."
  );

  /*!
   * Whether the isolated nodes of the compressed graph are continuously stored
   * at the end of the nodes array.
   */
#ifdef KAMINPAR_COMPRESSION_ISOLATED_NODES_SEPARATION
  static constexpr bool kIsolatedNodesSeparation = true;
#else
  static constexpr bool kIsolatedNodesSeparation = false;
#endif

  /**
   * Constructs a new CompressedNeighborhoods.
   *
   * @param nodes The nodes of the compressed neighborhoods.
   * @param compressed_edges The edges and edge weights of the compressed neighborhoods.
   * @param edge_weights The edge weights of the graph, which is only used when the graph has edge
   * weights and graph compression is disabled.
   * @param max_degree The maximum degree of the nodes.
   * @param num_edges The number of edges.
   * @param has_edge_weights Whether edge weights are stored
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
        _max_degree(max_degree),
        _num_edges(num_edges),
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
  }

  CompressedNeighborhoods(const CompressedNeighborhoods &) = delete;
  CompressedNeighborhoods &operator=(const CompressedNeighborhoods &) = delete;

  CompressedNeighborhoods(CompressedNeighborhoods &&) noexcept = default;
  CompressedNeighborhoods &operator=(CompressedNeighborhoods &&) noexcept = default;

  /**
   * Returns the maximum degree of the nodes.
   *
   * @return The maximum degree of the nodes.
   */
  [[nodiscard]] NodeID max_degree() const {
    return _max_degree;
  }

  /**
   * Returns the degree of a node.
   *
   * @param node The node whose degree is to be returned.
   * @return The degree of the node.
   */
  [[nodiscard]] NodeID degree(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) [[unlikely]] {
      return 0;
    }

    const auto header = decode_header(node, node_data, next_node_data);
    return header.degree;
  }

  /**
   * Returns incident edges of a nodes.
   *
   * @param node The node whose incident edges is to be returned.
   * @return The incident edges of the node.
   */
  [[nodiscard]] IotaRange<EdgeID> incident_edges(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) [[unlikely]] {
      return {0, 0};
    }

    const auto header = decode_header(node, node_data, next_node_data);
    return {header.first_edge, header.first_edge + header.degree};
  }

  /**
   * Decodes a neighborhood and invokes a caller with each adjacent node and corresponding edge
   * weight.
   *
   * @tparam kParallelDecoding Whether to decode the neighborhood in parallel.
   * @tparam Lambda The type of the caller to invoke.
   * @param u The node whose neighborhood is to be decoded.
   * @param l The caller to invoke.
   */
  template <bool kParallelDecoding = false, typename Lambda>
  void decode(const NodeID u, Lambda &&l) const {
    KASSERT(u < num_nodes());
    constexpr bool kInvokeDirectly = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;

    if (_has_edge_weights) [[unlikely]] {
      decode_neighborhood<true, kParallelDecoding>(u, std::forward<Lambda>(l));
    } else {
      if constexpr (kInvokeDirectly) {
        decode_neighborhood<false, kParallelDecoding>(u, [&](const EdgeID e, const NodeID v) {
          return l(e, v, 1);
        });
      } else {
        decode_neighborhood<false, kParallelDecoding>(u, [&](auto &&l2) {
          l([&](auto &&l3) { l2([&](const EdgeID e, const NodeID v) { return l3(e, v, 1); }); });
        });
      }
    }
  }

  /**
   * Decodes the leading edges of a neighborhood and invokes a caller with each adjacent node and
   * corresponding edge weight.
   *
   * @tparam Lambda The type of the caller to invoke.
   * @param u The node whose neighborhood is to be decoded.
   * @param max_num_neighbors The number of neighbors to decode.
   * @param l The caller to invoke.
   */
  template <typename Lambda>
  void decode(const NodeID u, const NodeID max_num_neighbors, Lambda &&l) const {
    KASSERT(u < num_nodes());
    KASSERT(max_num_neighbors > 0);

    static_assert(std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>);
    constexpr bool kNonStoppable =
        std::is_void_v<std::invoke_result_t<Lambda, EdgeID, NodeID, EdgeWeight>>;

    NodeID num_neighbors_visited = 1;
    const auto invoke_and_check = [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      bool abort = num_neighbors_visited++ >= max_num_neighbors;

      if constexpr (kNonStoppable) {
        l(e, v, w);
      } else {
        abort |= l(e, v, w);
      }

      return abort;
    };

    if (_has_edge_weights) [[unlikely]] {
      decode_neighborhood<true, false>(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
        return invoke_and_check(e, v, w);
      });
    } else {
      decode_neighborhood<false, false>(u, [&](const EdgeID e, const NodeID v) {
        return invoke_and_check(e, v, 1);
      });
    }
  }

  /**
   * Restricts the node array to a specific number of nodes.
   *
   * @param new_n The new number of nodes.
   */
  void restrict_nodes(const NodeID new_n) {
    _nodes.restrict(new_n);
  }

  /**
   * Unrestricts the node array.
   */
  void unrestrict_nodes() {
    _nodes.unrestrict();
  }

  /**
   * Returns the number of nodes.
   *
   * @return The number of nodes.
   */
  [[nodiscard]] EdgeID num_nodes() const {
    return _nodes.size() - 1;
  }

  /**
   * Returns the number of edges.
   *
   * @return The number of edges.
   */
  [[nodiscard]] EdgeID num_edges() const {
    return _num_edges;
  }

  /**
   * Returns whether the edges are weighted.
   *
   * @return Whether the edges are weighted.
   */
  [[nodiscard]] bool has_edge_weights() const {
    return _has_edge_weights;
  }

  /**
   * Returns the total edge weight.
   *
   * @return The total edge weight.
   */
  [[nodiscard]] EdgeWeight total_edge_weight() const {
    return _total_edge_weight;
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

  /**
   * Returns the used memory space in bytes.
   *
   * @return The used memory space in bytes.
   */
  [[nodiscard]] std::size_t memory_space() const {
    return _nodes.memory_space() + _compressed_edges.size() +
           _edge_weights.size() * sizeof(EdgeWeight);
  }

  /**
   * Returns ownership of the raw node array.
   *
   * @return Ownership of the raw node array.
   */
  [[nodiscard]] CompactStaticArray<EdgeID> &&take_raw_nodes() {
    return std::move(_nodes);
  }

  /**
   * Returns a reference to the raw node array.
   *
   * @return A reference to the raw node array.
   */
  [[nodiscard]] CompactStaticArray<EdgeID> &raw_nodes() {
    return _nodes;
  }

  /**
   * Returns a reference to the raw node array.
   *
   * @return A reference to the raw node array.
   */
  [[nodiscard]] const CompactStaticArray<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  /**
   * Returns a reference to the raw compressed edges.
   *
   * @return A reference to the raw compressed edges.
   */
  [[nodiscard]] const StaticArray<std::uint8_t> &raw_compressed_edges() const {
    return _compressed_edges;
  }

  /**
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

private:
  template <bool kHasEdgeWeights, bool kParallelDecoding, typename Lambda>
  void decode_neighborhood(const NodeID node, Lambda &&l) const {
    constexpr bool kInvokeDirectly = []() {
      if constexpr (kHasEdgeWeights) {
        return std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
      } else {
        return std::is_invocable_v<Lambda, EdgeID, NodeID>;
      }
    }();

    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) [[unlikely]] {
      return;
    }

    const auto header = decode_header(node, node_data, next_node_data);
    node_data += header.length;

    if constexpr (kHighDegreeEncoding) {
      if (header.degree >= kHighDegreeThreshold) {
        decode_parts<kHasEdgeWeights, kParallelDecoding>(
            node_data, node, header.degree, header.first_edge, std::forward<Lambda>(l)
        );
        return;
      }
    }

    invoke_indirect<kInvokeDirectly>(std::forward<Lambda>(l), [&](auto &&l2) {
      decode_edges<kHasEdgeWeights>(
          node_data,
          node,
          header.degree,
          header.first_edge,
          header.uses_intervals,
          std::forward<decltype(l2)>(l2)
      );
    });
  }

  [[nodiscard]] NeighborhoodHeader decode_header(
      const NodeID node,
      const std::uint8_t *const node_data,
      const std::uint8_t *const next_node_data
  ) const {
    const auto [first_edge, next_first_edge, uses_intervals, len] = [&] {
      if constexpr (kIntervalEncoding) {
        const auto [first_edge, uses_intervals, len] = marked_varint_decode<EdgeID>(node_data);
        const auto [next_first_edge, _, __] = marked_varint_decode<EdgeID>(next_node_data);

        return std::make_tuple(first_edge, next_first_edge, uses_intervals, len);
      } else {
        const auto [first_edge, len] = varint_decode<EdgeID>(node_data);
        const auto [next_first_edge, _] = varint_decode<EdgeID>(next_node_data);

        return std::make_tuple(first_edge, next_first_edge, false, len);
      }
    }();

    if constexpr (kIsolatedNodesSeparation) {
      const EdgeID ungapped_first_edge = first_edge + node;
      const NodeID degree = static_cast<NodeID>(1 + next_first_edge - first_edge);
      return {ungapped_first_edge, degree, uses_intervals, len};
    } else {
      const NodeID degree = static_cast<NodeID>(next_first_edge - first_edge);
      return {first_edge, degree, uses_intervals, len};
    }
  }

  template <bool kHasEdgeWeights, bool kParallelDecoding, typename Lambda>
  void decode_parts(
      const std::uint8_t *data,
      const NodeID node,
      const NodeID degree,
      const EdgeID edge,
      Lambda &&l
  ) const {
    constexpr bool kInvokeDirectly = []() {
      if constexpr (kHasEdgeWeights) {
        return std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
      } else {
        return std::is_invocable_v<Lambda, EdgeID, NodeID>;
      }
    }();

    const NodeID part_count = math::div_ceil(degree, kHighDegreePartLength);

    const auto iterate_part = [&](const NodeID part) {
      const NodeID part_offset = *((NodeID *)(data + sizeof(NodeID) * part));
      const std::uint8_t *part_data = data + part_offset;

      const NodeID part_count_m1 = part_count - 1;
      const bool last_part = part == part_count_m1;

      const EdgeID part_edge = edge + kHighDegreePartLength * part;
      const NodeID part_degree =
          last_part ? (degree - kHighDegreePartLength * part_count_m1) : kHighDegreePartLength;

      return invoke_indirect2<kInvokeDirectly, bool>(std::forward<Lambda>(l), [&](auto &&l2) {
        return decode_edges<kHasEdgeWeights>(
            part_data, node, part_degree, part_edge, true, std::forward<decltype(l2)>(l2)
        );
      });
    };

    if constexpr (kParallelDecoding) {
      tbb::parallel_for<NodeID>(0, part_count, iterate_part);
    } else {
      for (NodeID part = 0; part < part_count; ++part) {
        const bool stop = iterate_part(part);
        if (stop) {
          return;
        }
      }
    }
  }

  template <bool kHasEdgeWeights, typename Lambda>
  bool decode_edges(
      const std::uint8_t *data,
      const NodeID node,
      const NodeID degree,
      EdgeID edge,
      bool uses_intervals,
      Lambda &&l
  ) const {
    const EdgeID max_edge = edge + degree;
    EdgeWeight prev_edge_weight = 0;

    if constexpr (kIntervalEncoding) {
      if (uses_intervals) {
        const bool stop = decode_intervals<kHasEdgeWeights>(
            data, edge, prev_edge_weight, std::forward<Lambda>(l)
        );
        if (stop) {
          return true;
        }

        if (edge == max_edge) {
          return false;
        }
      }
    }

    return decode_gaps<kHasEdgeWeights>(
        data, node, edge, prev_edge_weight, max_edge, std::forward<Lambda>(l)
    );
  }

  template <bool kHasEdgeWeights, typename Lambda>
  bool decode_intervals(
      const std::uint8_t *&data, EdgeID &edge, EdgeWeight &prev_edge_weight, Lambda &&l
  ) const {
    using LambdaReturnType = std::conditional_t<
        kHasEdgeWeights,
        std::invoke_result<Lambda, EdgeID, NodeID, EdgeWeight>,
        std::invoke_result<Lambda, EdgeID, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    const auto invoke_caller = [&](const NodeID adjacent_node) {
      if constexpr (kHasEdgeWeights) {
        if constexpr (kCompressEdgeWeights) {
          const auto [edge_weight_gap, length] = signed_varint_decode<EdgeWeight>(data);
          data += length;

          const EdgeWeight edge_weight = edge_weight_gap + prev_edge_weight;
          prev_edge_weight = edge_weight;

          return l(edge, adjacent_node, edge_weight);
        } else {
          return l(edge, adjacent_node, _edge_weights[edge]);
        }
      } else {
        return l(edge, adjacent_node);
      }
    };

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
        if constexpr (kNonStoppable) {
          invoke_caller(cur_left_extreme + j);
        } else {
          const bool stop = invoke_caller(cur_left_extreme + j);
          if (stop) {
            return true;
          }
        }

        edge += 1;
      }
    }

    return false;
  }

  template <bool kHasEdgeWeights, typename Lambda>
  bool decode_gaps(
      const std::uint8_t *data,
      NodeID node,
      EdgeID &edge,
      EdgeWeight &prev_edge_weight,
      const EdgeID max_edge,
      Lambda &&l
  ) const {
    using LambdaReturnType = std::conditional_t<
        kHasEdgeWeights,
        std::invoke_result<Lambda, EdgeID, NodeID, EdgeWeight>,
        std::invoke_result<Lambda, EdgeID, NodeID>>::type;
    constexpr bool kNonStoppable = std::is_void_v<LambdaReturnType>;

    const auto invoke_caller = [&](const NodeID adjacent_node) {
      if constexpr (kHasEdgeWeights) {
        if constexpr (kCompressEdgeWeights) {
          const auto [edge_weight_gap, length] = signed_varint_decode<EdgeWeight>(data);
          data += length;

          const EdgeWeight edge_weight = edge_weight_gap + prev_edge_weight;
          prev_edge_weight = edge_weight;
          return l(edge, adjacent_node, edge_weight);
        } else {
          return l(edge, adjacent_node, _edge_weights[edge]);
        }
      } else {
        return l(edge, adjacent_node);
      }
    };

    const auto [first_gap, first_gap_len] = signed_varint_decode<SignedID>(data);
    data += first_gap_len;

    const NodeID first_adjacent_node = static_cast<NodeID>(first_gap + node);
    NodeID prev_adjacent_node = first_adjacent_node;

    if constexpr (kNonStoppable) {
      invoke_caller(first_adjacent_node);
    } else {
      const bool stop = invoke_caller(first_adjacent_node);
      if (stop) {
        return true;
      }
    }
    edge += 1;

    const auto handle_gap = [&](const NodeID gap) {
      const NodeID adjacent_node = gap + prev_adjacent_node + 1;
      prev_adjacent_node = adjacent_node;

      if constexpr (kNonStoppable) {
        invoke_caller(adjacent_node);
        edge += 1;
      } else {
        const bool stop = invoke_caller(adjacent_node);
        edge += 1;
        return stop;
      }
    };

    if constexpr (kRunLengthEncoding) {
      VarIntRunLengthDecoder<NodeID> rl_decoder(data, max_edge - edge);
      rl_decoder.decode(handle_gap);
    } else if constexpr (kStreamEncoding) {
      VarIntStreamDecoder<NodeID> sv_encoder(data, max_edge - edge);
      sv_encoder.decode(handle_gap);
    } else {
      while (edge != max_edge) {
        const auto [gap, gap_len] = varint_decode<NodeID>(data);
        data += gap_len;

        const NodeID adjacent_node = gap + prev_adjacent_node + 1;
        prev_adjacent_node = adjacent_node;

        if constexpr (kNonStoppable) {
          invoke_caller(adjacent_node);
        } else {
          const bool stop = invoke_caller(adjacent_node);
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
