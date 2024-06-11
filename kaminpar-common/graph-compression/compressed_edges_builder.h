#pragma once

#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph-compression/compressed_edges.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar {

template <typename NodeID, typename EdgeID, typename EdgeWeight> class CompressedEdgesBuilder {
  using CompressedEdges = kaminpar::CompressedEdges<NodeID, EdgeID>;
  using SignedID = CompressedEdges::SignedID;

  static constexpr bool kHighDegreeEncoding = CompressedEdges::kHighDegreeEncoding;
  static constexpr NodeID kHighDegreeThreshold = CompressedEdges::kHighDegreeThreshold;
  static constexpr NodeID kHighDegreePartLength = CompressedEdges::kHighDegreePartLength;
  static constexpr NodeID kIntervalEncoding = CompressedEdges::kIntervalEncoding;
  static constexpr NodeID kIntervalLengthTreshold = CompressedEdges::kIntervalLengthTreshold;
  static constexpr bool kRunLengthEncoding = CompressedEdges::kRunLengthEncoding;
  static constexpr bool kStreamEncoding = CompressedEdges::kStreamEncoding;
  static constexpr bool kIsolatedNodesSeparation = CompressedEdges::kIsolatedNodesSeparation;

  template <bool kActualNumEdges = true>
  [[nodiscard]] static std::size_t
  compressed_edge_array_max_size(const NodeID num_nodes, const EdgeID num_edges) {
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

    std::size_t max_size = num_nodes * edge_id_width + num_edges * varint_length(num_nodes);

    if constexpr (kHighDegreeEncoding) {
      if constexpr (kIntervalEncoding) {
        max_size += 2 * num_nodes * varint_max_length<NodeID>();
      } else {
        max_size += num_nodes * varint_max_length<NodeID>();
      }

      max_size += (num_edges / kHighDegreePartLength) * varint_max_length<NodeID>();
    }

    return max_size;
  }

public:
  /*!
   * Constructs a new CompressedEdgesBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_edge_weights Whether the graph to compress has edge weights.
   * @param edge_weights A reference to the edge weights of the compressed graph.
   */
  CompressedEdgesBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
      bool has_edge_weights,
      StaticArray<EdgeWeight> &edge_weights
  )
      : _has_edge_weights(has_edge_weights),
        _edge_weights(edge_weights) {
    const std::size_t max_size = compressed_edge_array_max_size(num_nodes, num_edges);
    _compressed_data_start = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
  }

  /*!
   * Constructs a new CompressedEdgesBuilder where the maxmimum degree specifies the number of edges
   * that are compressed at once.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param max_degree The maximum number of edges that are compressed at once.
   * @param has_edge_weights Whether the graph to compress has edge weights.
   * @param edge_weights A reference to the edge weights of the compressed graph.
   * @param edge_weights A reference to the edge weights of the compressed graph.
   */
  CompressedEdgesBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
      const NodeID max_degree,
      bool has_edge_weights,
      StaticArray<EdgeWeight> &edge_weights
  )
      : _has_edge_weights(has_edge_weights),
        _edge_weights(edge_weights) {
    const std::size_t max_size = compressed_edge_array_max_size<false>(num_nodes, max_degree);
    _compressed_data_start = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
  }

  CompressedEdgesBuilder(const CompressedEdgesBuilder &) = delete;
  CompressedEdgesBuilder &operator=(const CompressedEdgesBuilder &) = delete;

  CompressedEdgesBuilder(CompressedEdgesBuilder &&) noexcept = default;

  /*!
   * Initializes/resets the builder.
   *
   * @param first_edge The first edge ID of the first node to be added.
   */
  void init(const EdgeID first_edge) {
    _compressed_data = _compressed_data_start.get();

    _edge = first_edge;
    _max_degree = 0;
    _total_edge_weight = 0;

    _num_high_degree_nodes = 0;
    _num_high_degree_parts = 0;
    _num_interval_nodes = 0;
    _num_intervals = 0;
  }

  /*!
   * Adds the neighborhood of a node. Note that the neighbourhood vector is modified.
   *
   * @param node The node whose neighborhood to add.
   * @param neighbourhood The neighbourhood of the node to add.
   * @return The offset into the compressed edge array of the node.
   */
  EdgeID add(const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood) {
    // The offset into the compressed edge array of the start of the neighbourhood.
    const auto offset = static_cast<EdgeID>(_compressed_data - _compressed_data_start.get());

    const NodeID degree = neighbourhood.size();
    if (degree == 0) {
      return offset;
    }

    _max_degree = std::max(_max_degree, degree);

    // Store a pointer to the first byte of the first edge of this neighborhood. This byte encodes
    // in one of its bits whether interval encoding is used for this node, i.e., whether the nodes
    // has intervals in its neighbourhood.
    std::uint8_t *marked_byte = _compressed_data;

    // Store only the first edge for the source node. The degree can be obtained by determining the
    // difference between the first edge ids of a node and the next node. Additionally, store the
    // first edge as a gap when the isolated nodes are continuously stored at the end of the nodes
    // array.
    const EdgeID first_edge = _edge;
    if constexpr (kIntervalEncoding) {
      _compressed_data += marked_varint_encode(first_edge, false, _compressed_data);
    } else {
      _compressed_data += varint_encode(first_edge, _compressed_data);
    }

    // Only increment the edge if edge weights are not stored as otherwise the edge is
    // incremented with each edge weight being added.
    if (!_has_edge_weights) {
      _edge += degree;
    }

    // Sort the adjacent nodes in ascending order.
    std::sort(neighbourhood.begin(), neighbourhood.end(), [](const auto &a, const auto &b) {
      return a.first < b.first;
    });

    // If high-degree encoding is used then split the neighborhood if the degree crosses a
    // threshold. The neighborhood is split into equally sized parts (except possible the last part)
    // and each part is encoded independently. Furthermore, the offset at which the part is encoded
    // is also stored.
    if constexpr (kHighDegreeEncoding) {
      const bool split_neighbourhood = degree >= kHighDegreeThreshold;

      if (split_neighbourhood) {
        const NodeID part_count = math::div_ceil(degree, kHighDegreePartLength);
        const NodeID last_part_length = ((degree % kHighDegreePartLength) == 0)
                                            ? kHighDegreePartLength
                                            : (degree % kHighDegreePartLength);

        uint8_t *part_ptr = _compressed_data;
        _compressed_data += sizeof(NodeID) * part_count;

        for (NodeID i = 0; i < part_count; ++i) {
          const bool last_part = (i + 1) == part_count;
          const NodeID part_length = last_part ? last_part_length : kHighDegreePartLength;

          auto part_begin = neighbourhood.begin() + i * kHighDegreePartLength;
          auto part_end = part_begin + part_length;

          std::uint8_t *cur_part_ptr = part_ptr + sizeof(NodeID) * i;
          *((NodeID *)cur_part_ptr) = static_cast<NodeID>(_compressed_data - part_ptr);

          std::span<std::pair<NodeID, EdgeWeight>> part_neighbourhood(part_begin, part_end);
          add_edges(node, nullptr, part_neighbourhood);
        }

        _num_high_degree_nodes += 1;
        _num_high_degree_parts += part_count;
        return offset;
      }
    }

    add_edges(node, marked_byte, std::forward<decltype(neighbourhood)>(neighbourhood));
    return offset;
  }

  /*!
   * Returns the number of bytes that the compressed data of the added neighborhoods take up.
   *
   * @return The number of bytes that the compressed data of the added neighborhoods take up.
   */
  [[nodiscard]] std::size_t size() const {
    return static_cast<std::size_t>(_compressed_data - _compressed_data_start.get());
  }

  /*!
   * Returns a pointer to the start of the compressed data.
   *
   * @return A pointer to the start of the compressed data.
   */
  [[nodiscard]] const std::uint8_t *compressed_data() const {
    return _compressed_data_start.get();
  }

  /*!
   * Returns ownership of the compressed data
   *
   * @return Ownership of the compressed data.
   */
  [[nodiscard]] heap_profiler::unique_ptr<std::uint8_t> take_compressed_data() {
    return std::move(_compressed_data_start);
  }

  [[nodiscard]] std::size_t max_degree() const {
    return _max_degree;
  }

  [[nodiscard]] std::int64_t total_edge_weight() const {
    return _total_edge_weight;
  }

  [[nodiscard]] std::size_t num_high_degree_nodes() const {
    return _num_high_degree_nodes;
  }

  [[nodiscard]] std::size_t num_high_degree_parts() const {
    return _num_high_degree_parts;
  }

  [[nodiscard]] std::size_t num_interval_nodes() const {
    return _num_interval_nodes;
  }

  [[nodiscard]] std::size_t num_intervals() const {
    return _num_intervals;
  }

private:
  heap_profiler::unique_ptr<std::uint8_t> _compressed_data_start;
  std::uint8_t *_compressed_data;

  bool _has_edge_weights;
  StaticArray<EdgeWeight> &_edge_weights;

  EdgeID _edge;
  NodeID _max_degree;
  EdgeWeight _total_edge_weight;

  // Graph compression statistics
  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;

private:
  template <typename Container>
  void add_edges(const NodeID node, std::uint8_t *marked_byte, Container &&neighbourhood) {
    const auto store_edge_weight = [&](const EdgeWeight edge_weight) {
      _edge_weights[_edge++] = edge_weight;
      _total_edge_weight += edge_weight;
    };

    NodeID local_degree = neighbourhood.size();

    // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
    // least kIntervalLengthTreshold. Instead of storing all nodes, only encode the left extreme i
    // and the length j - i + 1. Left extremes are stored   static constexpr bool
    // kHighDegreeEncoding =  the differences between each left extreme and the previous right
    // extreme minus 2 (because there must be at least one integer between the end of an interval
    // and the beginning of the next one), except the first left extreme, which is stored directly.
    // The lengths are decremented by kIntervalLengthTreshold, the minimum length of an interval.
    if constexpr (kIntervalEncoding) {
      NodeID interval_count = 0;

      // Save the pointer to the interval count and skip the amount of bytes needed to store the
      // interval count as we can only determine the amount of intervals after finding all of
      // them.
      std::uint8_t *interval_count_ptr = _compressed_data;
      _compressed_data += sizeof(NodeID);

      if (local_degree >= kIntervalLengthTreshold) {
        NodeID interval_len = 1;
        NodeID previous_right_extreme = 2;
        NodeID prev_adjacent_node = (*neighbourhood.begin()).first;

        for (auto iter = neighbourhood.begin() + 1; iter != neighbourhood.end(); ++iter) {
          const NodeID adjacent_node = (*iter).first;

          if (prev_adjacent_node + 1 == adjacent_node) {
            interval_len++;

            // The interval ends if there are no more nodes or the next node is not the increment of
            // the current node.
            if (iter + 1 == neighbourhood.end() || (*(iter + 1)).first != adjacent_node + 1) {
              if (interval_len >= kIntervalLengthTreshold) {
                const NodeID left_extreme = adjacent_node + 1 - interval_len;
                const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
                const NodeID interval_length_gap = interval_len - kIntervalLengthTreshold;

                _compressed_data += varint_encode(left_extreme_gap, _compressed_data);
                _compressed_data += varint_encode(interval_length_gap, _compressed_data);

                for (NodeID i = 0; i < interval_len; ++i) {
                  std::pair<NodeID, EdgeWeight> &incident_edge = *(iter + 1 + i - interval_len);

                  // Set the adjacent node to a special value, which indicates for the gap encoder
                  // that the node has been encoded through an interval.
                  incident_edge.first = std::numeric_limits<NodeID>::max();

                  if (_has_edge_weights) {
                    store_edge_weight(incident_edge.second);
                  }
                }

                previous_right_extreme = adjacent_node;

                local_degree -= interval_len;
                interval_count += 1;
              }

              interval_len = 1;
            }
          }

          prev_adjacent_node = adjacent_node;
        }
      }

      // If intervals have been encoded store the interval count and set the bit in the marked byte
      // indicating that interval encoding has been used for the neighbourhood if the marked byte is
      // given. Otherwise, fix the amount of bytes stored as we don't store the interval count if no
      // intervals have been encoded.
      if (marked_byte == nullptr) {
        *((NodeID *)interval_count_ptr) = interval_count;
      } else if (interval_count > 0) {
        *((NodeID *)interval_count_ptr) = interval_count;
        *marked_byte |= 0b01000000;
      } else {
        _compressed_data -= sizeof(NodeID);
      }

      if (interval_count > 0) {
        _num_interval_nodes += 1;
        _num_intervals += interval_count;
      }

      // If all incident edges have been compressed   static constexpr bool kHighDegreeEncoding =
      // intervals then gap encoding cannot be applied.
      if (local_degree == 0) {
        return;
      }
    }

    // Store the remaining adjacent nodes   static constexpr bool kHighDegreeEncoding =  gap
    // encoding. That is instead of directly storing the nodes v_1, v_2, ..., v_{k - 1}, v_k, store
    // the gaps v_1 - u, v_2 - v_1 - 1, ..., v_k - v_{k - 1} - 1 between the nodes, where u is the
    // source node. Note that all gaps except the first one have to be positive as we sorted the
    // nodes in ascending order. Thus, only for the first gap the sign is additionally stored.
    auto iter = neighbourhood.begin();

    // Go to the first adjacent node that has not been encoded through an interval.
    if constexpr (kIntervalEncoding) {
      while ((*iter).first == std::numeric_limits<NodeID>::max()) {
        ++iter;
      }
    }

    const auto [first_adjacent_node, first_edge_weight] = *iter++;
    const SignedID first_gap = first_adjacent_node - static_cast<SignedID>(node);
    _compressed_data += signed_varint_encode(first_gap, _compressed_data);

    if (_has_edge_weights) {
      store_edge_weight(first_edge_weight);
    }

    VarIntRunLengthEncoder<NodeID> rl_encoder(_compressed_data);
    VarIntStreamEncoder<NodeID> sv_encoder(_compressed_data, local_degree - 1);

    NodeID prev_adjacent_node = first_adjacent_node;
    while (iter != neighbourhood.end()) {
      const auto [adjacent_node, edge_weight] = *iter++;

      // Skip the adjacent node since it has been encoded through an interval.
      if constexpr (kIntervalEncoding) {
        if (adjacent_node == std::numeric_limits<NodeID>::max()) {
          continue;
        }
      }

      const NodeID gap = adjacent_node - prev_adjacent_node - 1;
      if constexpr (kRunLengthEncoding) {
        _compressed_data += rl_encoder.add(gap);
      } else if constexpr (kStreamEncoding) {
        _compressed_data += sv_encoder.add(gap);
      } else {
        _compressed_data += varint_encode(gap, _compressed_data);
      }

      if (_has_edge_weights) {
        store_edge_weight(edge_weight);
      }

      prev_adjacent_node = adjacent_node;
    }

    if constexpr (kRunLengthEncoding) {
      rl_encoder.flush();
    } else if constexpr (kStreamEncoding) {
      sv_encoder.flush();
    }
  }
};

} // namespace kaminpar
