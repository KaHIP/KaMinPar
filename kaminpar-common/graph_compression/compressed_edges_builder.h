/*******************************************************************************
 * Compressed edges builder.
 *
 * @file:   compressed_edges_builder.h
 * @author: Daniel Salwasser
 * @date:   09.07.2024
 ******************************************************************************/
#pragma once

#include <limits>
#include <span>
#include <utility>

#include "kaminpar-common/graph_compression/compressed_neighborhoods.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar {

/*!
 * A builder to construct compressed edges.
 *
 * @tparam NodeID The type of integer to use to identify a node.
 * @tparam EdgeID The type of integer to use to identify an edge.
 * @tparam EdgeWeight The type of integer to use for edge weights.
 */
template <typename NodeID, typename EdgeID, typename EdgeWeight> class CompressedEdgesBuilder {
  using CompressedNeighborhoods = kaminpar::CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight>;

  static constexpr bool kCompressEdgeWeights = CompressedNeighborhoods::kCompressEdgeWeights;

  static constexpr bool kHighDegreeEncoding = CompressedNeighborhoods::kHighDegreeEncoding;
  static constexpr NodeID kHighDegreeThreshold = CompressedNeighborhoods::kHighDegreeThreshold;
  static constexpr NodeID kHighDegreePartLength = CompressedNeighborhoods::kHighDegreePartLength;

  static constexpr NodeID kIntervalEncoding = CompressedNeighborhoods::kIntervalEncoding;
  static constexpr NodeID kIntervalLengthTreshold =
      CompressedNeighborhoods::kIntervalLengthTreshold;

  static constexpr bool kRunLengthEncoding = CompressedNeighborhoods::kRunLengthEncoding;

  static constexpr bool kStreamVByteEncoding = CompressedNeighborhoods::kStreamVByteEncoding;
  static constexpr NodeID kStreamVByteThreshold = CompressedNeighborhoods::kStreamVByteThreshold;

  static constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();

  using SignedNodeID = std::int64_t;
  using SignedEdgeWeight = std::make_signed_t<EdgeWeight>;

  using StreamVByteGapEncoder =
      streamvbyte::StreamVByteEncoder<NodeID, streamvbyte::DifferentialCodingKind::D1>;

  using StreamVByteGapAndWeightEncoder =
      streamvbyte::StreamVByteEncoder<NodeID, streamvbyte::DifferentialCodingKind::D2>;

public:
  /*!
   * Returns the maximum size in bytes of the compressed edge array.
   *
   * @tparam kActualNumEdges Whether the number of edges given are of the whole graph instead of a
   * true subgraph.
   * @param num_nodes The number of nodes.
   * @param num_nodes The number of edges.
   * @param has_edge_weights Whether edge weights are stored.
   */
  template <bool kActualNumEdges = true>
  [[nodiscard]] static std::size_t compressed_edge_array_max_size(
      const NodeID num_nodes, const EdgeID num_edges, const bool has_edge_weights
  ) {
    std::size_t node_id_width = signed_varint_length(num_nodes);
    std::size_t edge_id_width;
    if constexpr (kActualNumEdges) {
      if constexpr (kIntervalEncoding) {
        edge_id_width = marked_varint_length(num_edges);
      } else {
        edge_id_width = varint_length(num_edges);
      }
    } else {
      edge_id_width = varint_max_length<EdgeID>();
    }

    std::size_t max_size = (num_nodes + 1) * edge_id_width + num_edges * node_id_width;

    if constexpr (kHighDegreeEncoding) {
      max_size += num_nodes * varint_max_length<NodeID>() +
                  (num_edges / kHighDegreePartLength) * varint_max_length<NodeID>();
    }

    if (kCompressEdgeWeights && has_edge_weights) {
      max_size += num_edges * varint_max_length<EdgeWeight>();
    }

    return max_size;
  }

  struct num_edges_ctor {};
  static constexpr num_edges_ctor num_edges_tag{};

  struct degree_ctor {};
  static constexpr degree_ctor degree_tag{};

  /*!
   * Constructs a new CompressedEdgesBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_edge_weights Whether the graph to compress has edge weights.
   * @param edge_weights A reference to the edge weights of the graph, which is only used when the
   * graph has edge weights and graph compression is disabled.
   */
  CompressedEdgesBuilder(
      num_edges_ctor,
      const NodeID num_nodes,
      const EdgeID num_edges,
      const bool has_edge_weights,
      StaticArray<EdgeWeight> &edge_weights
  )
      : _has_edge_weights(has_edge_weights),
        _edge_weights(edge_weights) {
    const std::size_t max_size =
        compressed_edge_array_max_size(num_nodes, num_edges, has_edge_weights);
    _compressed_edges = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
    _cur_compressed_edges = _compressed_edges.get();
    _compressed_data_max_size = 0;
  }

  /*!
   * Constructs a new CompressedEdgesBuilder where the maxmimum degree specifies the number
   * of edges that are compressed at once.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param max_degree The maximum number of edges that are compressed at once.
   * @param has_edge_weights Whether the graph to compress has edge weights.
   * @param edge_weights A reference to the edge weights of the graph, which is only used when the
   * graph has edge weights and graph compression is disabled.
   */
  CompressedEdgesBuilder(
      degree_ctor,
      const NodeID num_nodes,
      const NodeID max_degree,
      const bool has_edge_weights,
      StaticArray<EdgeWeight> &edge_weights
  )
      : _has_edge_weights(has_edge_weights),
        _edge_weights(edge_weights) {
    const std::size_t max_size =
        compressed_edge_array_max_size<false>(num_nodes, max_degree, has_edge_weights);
    _compressed_edges = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
    _cur_compressed_edges = _compressed_edges.get();
    _compressed_data_max_size = 0;
  }

  /*!
   * Destructs the CompressedEdgesBuilder and records the memory space of the compressed
   * edge array to the heap profiler if the data has not been taken.
   */
  ~CompressedEdgesBuilder() {
    if constexpr (kHeapProfiling) {
      if (_compressed_edges) {
        const auto prev_compressed_data_size = size();
        const std::size_t compressed_data_size =
            std::max(_compressed_data_max_size, prev_compressed_data_size);

        heap_profiler::HeapProfiler::global().record_alloc(
            _compressed_edges.get(), compressed_data_size
        );
      }
    }
  }

  CompressedEdgesBuilder(const CompressedEdgesBuilder &) = delete;
  CompressedEdgesBuilder &operator=(const CompressedEdgesBuilder &) = delete;

  CompressedEdgesBuilder(CompressedEdgesBuilder &&) noexcept = default;
  CompressedEdgesBuilder &operator=(CompressedEdgesBuilder &&) noexcept = delete;

  /*!
   * Initializes the builder.
   *
   * @param first_edge The first edge ID of the first node to be added.
   */
  void init(const EdgeID first_edge) {
    const auto prev_compressed_data_size = size();
    _compressed_data_max_size = std::max(_compressed_data_max_size, prev_compressed_data_size);
    _cur_compressed_edges = _compressed_edges.get();

    _cur_edge = first_edge;
    _max_degree = 0;
    _total_edge_weight = 0;
    _cur_edge_weight = first_edge;

    _num_high_degree_nodes = 0;
    _num_high_degree_parts = 0;
    _num_interval_nodes = 0;
    _num_intervals = 0;
  }

  /*!
   * Adds the (possibly weighted) neighborhood of a node. Note that the neighbourhood vector is
   * modified.
   *
   * @param node The node whose neighborhood to add.
   * @param neighbourhood The neighbourhood of the node to add.
   * @return The offset into the compressed edge array of the node.
   */
  template <typename Container> EdgeID add(const NodeID node, Container &neighborhood) {
    using Neighbor = std::remove_reference_t<Container>::value_type;
    constexpr bool kIsNeighbor = std::is_same_v<Neighbor, NodeID>;
    constexpr bool kIsWeightedNeighbor = std::is_same_v<Neighbor, std::pair<NodeID, EdgeWeight>>;
    static_assert(kIsNeighbor || kIsWeightedNeighbor);

    const EdgeID offset = current_offset();
    NodeID degree = neighborhood.size();
    if (degree == 0) [[unlikely]] {
      return offset;
    }

    if constexpr (kIsWeightedNeighbor) {
      std::sort(neighborhood.begin(), neighborhood.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
      });
    } else {
      std::sort(neighborhood.begin(), neighborhood.end());
    }

    NodeID num_intervals;
    if constexpr (kIntervalEncoding) {
      bool has_intervals;
      if (kHighDegreeEncoding && degree >= kHighDegreeThreshold) {
        has_intervals = false;
      } else {
        num_intervals = count_intervals(neighborhood);
        has_intervals = num_intervals > 0;
        _num_interval_nodes += has_intervals ? 1 : 0;
      }

      marked_varint_encode(_cur_edge, has_intervals, &_cur_compressed_edges);
    } else {
      varint_encode(_cur_edge, &_cur_compressed_edges);
    }

    _cur_edge += degree;

    if constexpr (kHighDegreeEncoding) {
      const bool split_neighbourhood = degree >= kHighDegreeThreshold;

      if (split_neighbourhood) {
        const NodeID num_parts = math::div_ceil(degree, kHighDegreePartLength);
        const NodeID last_part_length = math::mod_ceil(degree, kHighDegreePartLength);

        std::uint8_t *part_ptr = _cur_compressed_edges;
        _cur_compressed_edges += sizeof(NodeID) * num_parts;

        bool has_intervals = false;
        for (NodeID i = 0; i < num_parts; ++i) {
          const bool last_part = (i + 1) == num_parts;
          const NodeID part_length = last_part ? last_part_length : kHighDegreePartLength;

          auto part_begin = neighborhood.begin() + i * kHighDegreePartLength;
          auto part_end = part_begin + part_length;
          auto part_neighborhood = std::span<Neighbor>(part_begin, part_end);

          NodeID *cur_part_ptr = reinterpret_cast<NodeID *>(part_ptr) + i;
          *cur_part_ptr = static_cast<NodeID>(_cur_compressed_edges - part_ptr);

          NodeID num_intervals;
          if constexpr (kIntervalEncoding) {
            num_intervals = count_intervals(part_neighborhood);

            if (num_intervals > 0) {
              *cur_part_ptr |= math::kSetMSB<NodeID>;
              has_intervals = true;
            }
          }

          add_edges(node, num_intervals, part_neighborhood);
        }

        _num_high_degree_nodes += 1;
        _num_high_degree_parts += num_parts;
        _num_interval_nodes += has_intervals ? 1 : 0;
        return offset;
      }
    }

    add_edges(node, num_intervals, neighborhood);
    return offset;
  }

  /*!
   * Returns the number of bytes that the compressed data of the added neighborhoods take up.
   *
   * @return The number of bytes that the compressed data of the added neighborhoods take up.
   */
  [[nodiscard]] std::size_t size() const {
    return static_cast<std::size_t>(current_offset());
  }

  /*!
   * Returns a pointer to the start of the compressed data.
   *
   * @return A pointer to the start of the compressed data.
   */
  [[nodiscard]] const std::uint8_t *compressed_data() const {
    return _compressed_edges.get();
  }

  /*!
   * Returns ownership of the compressed data
   *
   * @return Ownership of the compressed data.
   */
  [[nodiscard]] heap_profiler::unique_ptr<std::uint8_t> take_compressed_data() {
    return std::move(_compressed_edges);
  }

  /*!
   * Returns the maximum degree.
   *
   * @return The maximum degree.
   */
  [[nodiscard]] std::size_t max_degree() const {
    return _max_degree;
  }

  /*!
   * Returns the total edge weight.
   *
   * @return The total edge weight.
   */
  [[nodiscard]] std::int64_t total_edge_weight() const {
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

private:
  [[nodiscard]] std::uint64_t current_offset() const {
    return static_cast<std::uint64_t>(_cur_compressed_edges - _compressed_edges.get());
  }

  template <typename Container>
  static void
  set_adjacent_node(Container &neighborhood, const NodeID num_neighbor, const NodeID value) {
    using Neighbor = std::remove_reference_t<Container>::value_type;
    constexpr bool kIsWeightedNeighbor = std::is_same_v<Neighbor, std::pair<NodeID, EdgeWeight>>;

    if constexpr (kIsWeightedNeighbor) {
      neighborhood[num_neighbor].first = value;
    } else {
      neighborhood[num_neighbor] = value;
    }
  }

  template <typename Container>
  [[nodiscard]] static NodeID
  get_adjacent_node(const Container &neighborhood, const NodeID num_neighbor) {
    using Neighbor = std::remove_reference_t<Container>::value_type;
    constexpr bool kIsWeightedNeighbor = std::is_same_v<Neighbor, std::pair<NodeID, EdgeWeight>>;

    if constexpr (kIsWeightedNeighbor) {
      return neighborhood[num_neighbor].first;
    } else {
      return neighborhood[num_neighbor];
    }
  }

  template <typename Container>
  [[nodiscard]] static EdgeWeight
  get_edge_weight(const Container &neighborhood, const NodeID num_neighbor) {
    using Neighbor = std::remove_reference_t<Container>::value_type;
    constexpr bool kIsWeightedNeighbor = std::is_same_v<Neighbor, std::pair<NodeID, EdgeWeight>>;
    static_assert(kIsWeightedNeighbor);

    return neighborhood[num_neighbor].second;
  }

  void encode_edge_weight(const EdgeWeight edge_weight, EdgeWeight &prev_edge_weight) {
    if (!_has_edge_weights) {
      return;
    }

    _total_edge_weight += edge_weight;

    if constexpr (kCompressEdgeWeights) {
      const SignedEdgeWeight edge_weight_gap =
          edge_weight - static_cast<SignedEdgeWeight>(prev_edge_weight);

      signed_varint_encode(edge_weight_gap, &_cur_compressed_edges);
      prev_edge_weight = edge_weight;
    } else {
      _edge_weights[_cur_edge_weight++] = edge_weight;
    }
  }

  template <typename Container>
  void add_edges(const NodeID node, const NodeID num_intervals, Container &neighborhood) {
    NodeID degree = neighborhood.size();
    EdgeWeight prev_edge_weight = 0;

    if constexpr (kIntervalEncoding) {
      const NodeID num_remaining_nodes =
          encode_intervals(num_intervals, prev_edge_weight, neighborhood);
      degree = num_remaining_nodes;
    }

    encode_gaps(node, degree, prev_edge_weight, neighborhood);
  }

  template <bool kInvalidate = false, typename Container, typename Lambda>
  void parse_intervals(const Container &neighborhood, Lambda &&l) const {
    const NodeID degree = neighborhood.size();
    if (degree < kIntervalLengthTreshold) {
      return;
    }

    NodeID interval_len = 1;
    NodeID prev_adjacent_node = get_adjacent_node(neighborhood, 0);
    for (NodeID i = 1; i < degree; ++i) {
      const NodeID adjacent_node = get_adjacent_node(neighborhood, i);

      const bool not_successive_increment = prev_adjacent_node + 1 != adjacent_node;
      prev_adjacent_node = adjacent_node;
      if (not_successive_increment) {
        continue;
      }

      interval_len += 1;
      if ((i + 1 < degree) && (adjacent_node + 1 == get_adjacent_node(neighborhood, i + 1))) {
        continue;
      }

      if (interval_len >= kIntervalLengthTreshold) {
        const NodeID right_extreme = adjacent_node;
        const NodeID left_extreme = right_extreme - (interval_len - 1);
        l(left_extreme, right_extreme, interval_len, i - (interval_len - 1));
      }

      interval_len = 1;
    }
  }

  template <typename Container>
  [[nodiscard]] NodeID count_intervals(const Container &neighborhood) const {
    NodeID num_intervals = 0;

    parse_intervals(neighborhood, [&](const NodeID, const NodeID, const NodeID, const NodeID) {
      num_intervals += 1;
    });

    return num_intervals;
  }

  template <typename Container>
  NodeID encode_intervals(
      const NodeID num_intervals, EdgeWeight &prev_edge_weight, Container &neighborhood
  ) {
    using Neighbor = std::remove_reference_t<Container>::value_type;
    constexpr bool kHasEdgeWeights = std::is_same_v<Neighbor, std::pair<NodeID, EdgeWeight>>;

    NodeID num_remaining_nodes = neighborhood.size();
    if (num_intervals > 0) {
      varint_encode(num_intervals - 1, &_cur_compressed_edges);
      _num_intervals += num_intervals;

      NodeID prev_right_extreme = 0;
      parse_intervals(
          neighborhood,
          [&](const NodeID left_extreme,
              const NodeID right_extreme,
              const NodeID interval_len,
              const NodeID index) {
            const NodeID left_extreme_gap = left_extreme - prev_right_extreme;
            const NodeID interval_len_gap = interval_len - kIntervalLengthTreshold;

            varint_encode(left_extreme_gap, &_cur_compressed_edges);
            varint_encode(interval_len_gap, &_cur_compressed_edges);

            prev_right_extreme = right_extreme + 2;
            num_remaining_nodes -= interval_len;
            for (NodeID i = 0; i < interval_len; ++i) {
              const NodeID pos = index + i;

              // Set the adjacent node to a special value, which indicates to the gap encoder
              // that the node has been encoded through an interval.
              set_adjacent_node(neighborhood, pos, kInvalidNodeID);

              if constexpr (kHasEdgeWeights) {
                const EdgeWeight edge_weight = get_edge_weight(neighborhood, pos);
                encode_edge_weight(edge_weight, prev_edge_weight);
              }
            }
          }
      );
    }

    return num_remaining_nodes;
  }

  template <typename Container>
  void encode_gaps(
      const NodeID node, const NodeID degree, EdgeWeight &prev_edge_weight, Container &neighborhood
  ) {
    using Neighbor = std::remove_reference_t<Container>::value_type;
    constexpr bool kHasEdgeWeights = std::is_same_v<Neighbor, std::pair<NodeID, EdgeWeight>>;

    if (degree == 0) {
      return;
    }

    NodeID i = 0;
    while (get_adjacent_node(neighborhood, i) == kInvalidNodeID) {
      i += 1;
    }

    const NodeID first_adjacent_node = get_adjacent_node(neighborhood, i);
    const SignedNodeID first_gap = first_adjacent_node - static_cast<SignedNodeID>(node);
    signed_varint_encode(first_gap, &_cur_compressed_edges);
    if constexpr (kHasEdgeWeights) {
      const EdgeWeight edge_weight = get_edge_weight(neighborhood, i);
      encode_edge_weight(edge_weight, prev_edge_weight);
    }

    i += 1;

    if constexpr (kRunLengthEncoding) {
      VarIntRunLengthEncoder<NodeID> rl_encoder(_cur_compressed_edges);

      NodeID prev_adjacent_node = first_adjacent_node;
      while (i < neighborhood.size()) {
        const NodeID adjacent_node = get_adjacent_node(neighborhood, i);
        if (adjacent_node == kInvalidNodeID) {
          i += 1;
          continue;
        }

        const NodeID gap = adjacent_node - prev_adjacent_node - 1;
        prev_adjacent_node = adjacent_node;

        _cur_compressed_edges += rl_encoder.add(gap);
        if constexpr (kHasEdgeWeights) {
          const EdgeWeight edge_weight = get_edge_weight(neighborhood, i);
          encode_edge_weight(edge_weight, prev_edge_weight);
        }

        i += 1;
      }

      rl_encoder.flush();
      return;
    } else if constexpr (kStreamVByteEncoding) {
      const NodeID num_remaining_gaps = degree - 1;

      if (num_remaining_gaps >= kStreamVByteThreshold) [[likely]] {
        if constexpr (kHasEdgeWeights) {
          if (_has_edge_weights) {
            StreamVByteGapAndWeightEncoder encoder(num_remaining_gaps * 2, _cur_compressed_edges);

            while (i < neighborhood.size()) {
              const NodeID adjacent_node = get_adjacent_node(neighborhood, i);
              if (adjacent_node == kInvalidNodeID) {
                i += 1;
                continue;
              }

              const EdgeWeight weight = get_edge_weight(neighborhood, i);
              _cur_compressed_edges += encoder.add(adjacent_node);
              _cur_compressed_edges += encoder.add(weight);

              i += 1;
            }

            encoder.flush();
            return;
          }
        }

        StreamVByteGapEncoder encoder(num_remaining_gaps, _cur_compressed_edges);
        while (i < neighborhood.size()) {
          const NodeID adjacent_node = get_adjacent_node(neighborhood, i++);
          if (adjacent_node == kInvalidNodeID) {
            continue;
          }

          _cur_compressed_edges += encoder.add(adjacent_node);
        }

        encoder.flush();
        return;
      }
    }

    NodeID prev_adjacent_node = first_adjacent_node;
    while (i < neighborhood.size()) {
      const NodeID adjacent_node = get_adjacent_node(neighborhood, i);
      if (adjacent_node == kInvalidNodeID) {
        i += 1;
        continue;
      }

      const NodeID gap = adjacent_node - prev_adjacent_node - 1;
      prev_adjacent_node = adjacent_node;

      varint_encode(gap, &_cur_compressed_edges);
      if constexpr (kHasEdgeWeights) {
        const EdgeWeight edge_weight = get_edge_weight(neighborhood, i);
        encode_edge_weight(edge_weight, prev_edge_weight);
      }

      i += 1;
    }
  }

private:
  heap_profiler::unique_ptr<std::uint8_t> _compressed_edges;
  std::uint8_t *_cur_compressed_edges;
  std::size_t _compressed_data_max_size;

  bool _has_edge_weights;
  EdgeWeight _total_edge_weight;
  EdgeID _cur_edge_weight;
  StaticArray<EdgeWeight> &_edge_weights;

  EdgeID _cur_edge;
  NodeID _max_degree;

  // Graph compression statistics
  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;
};

} // namespace kaminpar
