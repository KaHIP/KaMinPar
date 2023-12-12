/*******************************************************************************
 * Compressed static graph representations.
 *
 * @file:   compressed_graph.h
 * @author: Daniel Salwasser
 * @date:   07.11.2023
 ******************************************************************************/
#pragma once

#include <utility>
#include <vector>

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
   * The minimum degree of a node to be considered high degree.
   */
  static constexpr NodeID kHighDegreeThreshold = 10000;

  /*!
   * Whether interval encoding is used.
   */
  static constexpr bool kIntervalEncoding = true;

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr NodeID kIntervalLengthTreshold = 3;

  /*!
   * Constructs a new compressed graph.
   *
   * @param nodes The node array which stores for each node the offset in the compressed edges array
   * of the first edge.
   * @param compressed_edges The edge array which stores the edges for each node in a compressed
   * format.
   * @param node_weights The array of node weights in which the weights of each node in the
   * respective entry are stored.
   * @param edge_weights The array of edge weights in which the weights of each edge in the
   * respective entry are stored.
   * @param edge_count The number of edges stored in the compressed edge array.
   * @param high_degree_count The number of nodes which have high degree.
   * @param part_count The number of parts that result from splitting the neighbourhood of high
   * degree nodes.
   * @param interval_count The number of nodes/parts which use interval encoding.
   */
  explicit CompressedGraph(
      StaticArray<EdgeID> nodes,
      StaticArray<std::uint8_t> compressed_edges,
      StaticArray<NodeWeight> node_weights,
      StaticArray<EdgeWeight> edge_weights,
      EdgeID edge_count,
      std::size_t high_degree_count,
      std::size_t part_count,
      std::size_t interval_count
  );

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
    return _node_count;
  };

  [[nodiscard]] EdgeID m() const final {
    return _edge_count;
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
  [[nodiscard]] inline NodeID degree(const NodeID node) const final {
    const std::uint8_t *data = _compressed_edges.data() + _nodes[node];
    auto [degree, _] = varint_decode<NodeID>(data);
    return degree;
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
    return IotaRange(static_cast<EdgeID>(0), m());
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data() + _nodes[node];

    const auto [degree, degree_len] = varint_decode<NodeID>(data);
    data += degree_len;

    if constexpr (kIntervalEncoding) {
      const bool split_neighbourhood = degree > kHighDegreeThreshold;

      if (!split_neighbourhood) {
        const auto [first_edge, _, __] = marked_varint_decode<EdgeID>(data);
        return IotaRange<EdgeID>(first_edge, first_edge + degree);
      } else {
        const auto [first_edge, _] = varint_decode<EdgeID>(data);
        return IotaRange<EdgeID>(first_edge, first_edge + degree);
      }
    } else {
      const auto [first_edge, _] = varint_decode<EdgeID>(data);
      return IotaRange<EdgeID>(first_edge, first_edge + degree);
    }
  }

  template <typename Lambda> inline void adjacent_nodes(const NodeID node, Lambda &&l) const {
    iterate_neighborhood(node, [&](EdgeID incident_edge, NodeID adjacent_node) {
      l(adjacent_node);
    });
  }

  template <typename Lambda> inline void neighbors(const NodeID node, Lambda &&l) const {
    iterate_neighborhood(node, std::forward<Lambda>(l));
  }

  template <typename Lambda>
  inline void neighbors(const NodeID node, const NodeID max_neighbor_count, Lambda &&l) const {
    iterate_neighborhood<true>(node, std::forward<Lambda>(l), max_neighbor_count);
  }

  template <typename Lambda>
  inline void pfor_neighbors(const NodeID node, const NodeID max_neighbor_count, Lambda &&l) const {
    iterate_neighborhood<true, true>(node, std::forward<Lambda>(l), max_neighbor_count);
  }

  // Graph permutation
  inline void set_permutation(StaticArray<NodeID> permutation) final {
    _permutation = std::move(permutation);
  }

  [[nodiscard]] inline bool permuted() const final {
    return !_permutation.empty();
  }

  [[nodiscard]] inline NodeID map_original_node(const NodeID node) const final {
    return _permutation[node];
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

  void update_total_node_weight() final;

  // Compressions statistics

  /**
   * Returns the number of nodes which have high degree.
   *
   * @returns The number of nodes which have high degree.
   */
  [[nodiscard]] std::size_t high_degree_count() const {
    return _high_degree_count;
  }

  /**
   * Returns the number of parts that result from splitting the neighborhood of high degree nodes.
   *
   * @returns The number of parts that result from splitting the neighborhood of high degree nodes.
   */
  [[nodiscard]] std::size_t part_count() const {
    return _part_count;
  }

  /**
   * Returns the number of nodes/parts which use interval encoding.
   *
   * @returns The number of nodes/parts which use interval encoding.
   */
  [[nodiscard]] std::size_t interval_count() const {
    return _interval_count;
  }

  /*!
   * Returns the amount of memory in bytes used by the data structure.
   *
   * @return The amount of memory in bytes used by the data structure.
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

  const NodeID _node_count;
  const EdgeID _edge_count;

  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;
  NodeWeight _max_node_weight = kInvalidNodeWeight;

  StaticArray<NodeID> _permutation;

  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;

  const std::size_t _high_degree_count;
  const std::size_t _part_count;
  const std::size_t _interval_count;

  StaticArray<NodeID> _raw_edges_dummy{};
  StaticArray<EdgeWeight> _raw_edge_weights_dummy{};

  void init_degree_buckets();

  template <bool max_edges = false, bool parallel = false, typename Lambda>
  inline void iterate_neighborhood(
      const NodeID node, Lambda &&l, NodeID max_neighbor_count = std::numeric_limits<NodeID>::max()
  ) const {
    const std::uint8_t *data = _compressed_edges.data() + _nodes[node];
    const std::uint8_t *end = _compressed_edges.data() + _nodes[node + 1];

    const auto [degree, degree_len] = varint_decode<NodeID>(data);
    data += degree_len;

    if constexpr (max_edges) {
      max_neighbor_count = std::min(degree, max_neighbor_count);
    }

    const bool split_neighbourhood = degree > kHighDegreeThreshold;
    if (split_neighbourhood) {
      const auto [first_edge, first_edge_len] = varint_decode<EdgeID>(data);
      data += first_edge_len;

      const NodeID part_count = ((degree % kHighDegreeThreshold) == 0)
                                    ? (degree / kHighDegreeThreshold)
                                    : ((degree / kHighDegreeThreshold) + 1);

      const NodeID max_part_count = std::min(
          part_count,
          ((max_neighbor_count % kHighDegreeThreshold) == 0)
              ? (max_neighbor_count / kHighDegreeThreshold)
              : ((max_neighbor_count / kHighDegreeThreshold) + 1)
      );

      const NodeID max_neighbor_rem = ((max_neighbor_count % kHighDegreeThreshold) == 0)
                                          ? kHighDegreeThreshold
                                          : (max_neighbor_count % kHighDegreeThreshold);

      const auto iterate_part = [&](const NodeID part) {
        const std::uint8_t *part_data = data + *((NodeID *)(data + sizeof(NodeID) * part));
        const EdgeID part_first_edge = first_edge + kHighDegreeThreshold * part;

        const bool last_part = part + 1 == max_part_count;
        if (last_part) {
          iterate_edges<max_edges>(
              part_data, end, node, max_neighbor_rem, part_first_edge, true, std::forward<Lambda>(l)
          );
        } else {
          const std::uint8_t *part_end = (data + *((NodeID *)(data + sizeof(NodeID) * (part + 1))));
          iterate_edges<false>(
              part_data, part_end, node, 0, part_first_edge, true, std::forward<Lambda>(l)
          );
        }
      };

      if constexpr (parallel) {
        tbb::parallel_for<NodeID>(
            0, max_part_count, std::forward<decltype(iterate_part)>(iterate_part)
        );
      } else {
        for (NodeID part = 0; part < max_part_count; ++part) {
          iterate_part(part);
        }
      }
    } else {
      const auto [first_edge, uses_intervals] = [&] {
        if constexpr (kIntervalEncoding) {
          auto [first_edge, marker_set, first_edge_len] = marked_varint_decode<EdgeID>(data);
          data += first_edge_len;
          return std::make_pair(first_edge, marker_set);
        } else {
          auto [first_edge, first_edge_len] = varint_decode<EdgeID>(data);
          data += first_edge_len;
          return std::make_pair(first_edge, false);
        }
      }();

      iterate_edges<max_edges>(
          data, end, node, max_neighbor_count, first_edge, uses_intervals, std::forward<Lambda>(l)
      );
    }
  }

  template <bool max_edges = false, typename Lambda>
  inline void iterate_edges(
      const std::uint8_t *data,
      const std::uint8_t *end,
      const NodeID node,
      const NodeID max_neighbor_count,
      const NodeID first_edge,
      const bool uses_intervals,
      Lambda &&l
  ) const {
    EdgeID edge = first_edge;
    if constexpr (max_edges) {
      if (edge - first_edge == max_neighbor_count) {
        return;
      }
    }

    if constexpr (kIntervalEncoding) {
      if (uses_intervals) {
        const NodeID interval_count = *((NodeID *)data);
        data += sizeof(NodeID);

        NodeID previous_right_extreme = 2;
        for (NodeID i = 0; i < interval_count; i++) {
          const auto [left_extreme_gap, left_extreme_gap_len] = varint_decode<NodeID>(data);
          data += left_extreme_gap_len;

          const auto [interval_length_gap, interval_length_gap_len] = varint_decode<NodeID>(data);
          data += interval_length_gap_len;

          const NodeID cur_left_extreme = left_extreme_gap + previous_right_extreme - 2;
          const NodeID cur_interval_len = interval_length_gap + kIntervalLengthTreshold;
          previous_right_extreme = cur_left_extreme + cur_interval_len - 1;

          for (NodeID i = 0; i < cur_interval_len; i++) {
            l(edge++, cur_left_extreme + i);

            if constexpr (max_edges) {
              if (edge - first_edge == max_neighbor_count) {
                return;
              }
            }
          }
        }
      }
    }

    if constexpr (!max_edges) {
      if (data == end) {
        return;
      }
    }

    const auto [first_gap, first_gap_len] = signed_varint_decode<NodeID>(data);
    data += first_gap_len;

    const NodeID first_adjacent_node = first_gap + node;
    NodeID prev_adjacent_node = first_adjacent_node;

    l(edge++, first_adjacent_node);

    const auto iterate_gap = [&]() {
      const auto [gap, gap_len] = varint_decode<NodeID>(data);
      data += gap_len;

      const NodeID adjacent_node = gap + prev_adjacent_node;
      prev_adjacent_node = adjacent_node;

      l(edge++, adjacent_node);
    };

    if constexpr (max_edges) {
      while (edge - first_edge < max_neighbor_count) {
        iterate_gap();
      }
    } else {
      while (data != end) {
        iterate_gap();
      }
    }
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
   * Adds a node to the compressed graph, modifying the neighbourhood vector.
   *
   * @param node The node to add.
   * @param neighbourhood The neighbourhood of the node to add, i.e. the adjacent nodes and the edge
   * weight.
   */
  void add_node(const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood);

  /*!
   * Sets the weight of a node.
   *
   * @param node The node whose weight is to be set.
   * @param weight The weight to be set.
   */
  void set_node_weight(const NodeID node, const NodeWeight weight);

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

  bool _store_node_weights;
  bool _store_edge_weights;
  std::int64_t _total_node_weight;
  std::int64_t _total_edge_weight;

  std::uint8_t *_compressed_edges;
  std::uint8_t *_cur_compressed_edges;

  EdgeID _edge_count;
  std::size_t _high_degree_count;
  std::size_t _part_count;
  std::size_t _interval_count;

  void add_edges(
      NodeID node,
      std::uint8_t *marked_byte,
      std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
  );
};

} // namespace kaminpar::shm
