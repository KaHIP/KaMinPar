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

#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/ranges.h"
#include "kaminpar-common/varint_codec.h"
#include "kaminpar-common/varint_run_length_codec.h"
#include "kaminpar-common/varint_stream_codec.h"

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
   * The length of a part when splitting the neighbourhood of a high degree node.
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
      "Either run-length or stream encoding can be used for varints but not both."
  );

#ifdef KAMINPAR_COMPRESSION_ISOLATED_NODES_SEPARATION
  /*!
   * Whether the isolated nodes of the compressed graph are continuously stored at the end of the
   * nodes array.
   */
  static constexpr bool kIsolatedNodesSeparation = true;
#else
  /*!
   * Whether the isolated nodes of the compressed graph are continuously stored at the end of the
   * nodes array.
   */
  static constexpr bool kIsolatedNodesSeparation = false;
#endif

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
   * @param max_degree The maximum degree of the graph.
   * @param sorted Whether the nodes are stored by deg-buckets order.
   * @param high_degree_count The number of nodes which have high degree.
   * @param part_count The number of parts that result from splitting the neighbourhood of high
   * degree nodes.
   * @param interval_count The number of nodes/parts which use interval encoding.
   */
  explicit CompressedGraph(
      CompactStaticArray<EdgeID> nodes,
      StaticArray<std::uint8_t> compressed_edges,
      StaticArray<NodeWeight> node_weights,
      StaticArray<EdgeWeight> edge_weights,
      EdgeID edge_count,
      NodeID max_degree,
      bool sorted,
      std::size_t high_degree_count,
      std::size_t part_count,
      std::size_t interval_count
  );

  CompressedGraph(const CompressedGraph &) = delete;
  CompressedGraph &operator=(const CompressedGraph &) = delete;

  CompressedGraph(CompressedGraph &&) noexcept = default;
  CompressedGraph &operator=(CompressedGraph &&) noexcept = default;

  // Direct member access -- used for some "low level" operations
  [[nodiscard]] inline CompactStaticArray<EdgeID> &raw_nodes() {
    return _nodes;
  }

  [[nodiscard]] inline const CompactStaticArray<EdgeID> &raw_nodes() const {
    return _nodes;
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() {
    return _node_weights;
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const {
    return _node_weights;
  }

  [[nodiscard]] inline CompactStaticArray<EdgeID> &&take_raw_nodes() {
    return std::move(_nodes);
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() {
    return std::move(_node_weights);
  }

  [[nodiscard]] const StaticArray<std::uint8_t> &raw_compressed_edges() const {
    return _compressed_edges;
  }

  [[nodiscard]] const StaticArray<EdgeWeight> &raw_edge_weights() const {
    return _edge_weights;
  }

  // Size of the graph
  [[nodiscard]] NodeID n() const final {
    return static_cast<NodeID>(_nodes.size() - 1);
  };

  [[nodiscard]] EdgeID m() const final {
    return _edge_count;
  }

  // Node and edge weights
  [[nodiscard]] inline bool node_weighted() const final {
    return static_cast<NodeWeight>(n()) != total_node_weight();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const final {
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
    return edge_weighted() ? _edge_weights[e] : 1;
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _total_edge_weight;
  }

  // Low-level access to the graph structure
  [[nodiscard]] inline NodeID max_degree() const final {
    return _max_degree;
  }

  [[nodiscard]] inline NodeID degree(const NodeID node) const final {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return 0;
    }

    const auto [first_edge, degree, _, __] = decode_header(node, node_data, next_node_data);
    return degree;
  }

  // Iterators for nodes / edges
  [[nodiscard]] IotaRange<NodeID> nodes() const final {
    return IotaRange(static_cast<NodeID>(0), n());
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return IotaRange(static_cast<EdgeID>(0), m());
  }

  // Parallel iteration
  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  // Graph operations
  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return IotaRange<EdgeID>(0, 0);
    }

    const auto [first_edge, degree, _, __] = decode_header(node, node_data, next_node_data);
    return IotaRange<EdgeID>(first_edge, first_edge + degree);
  }

  template <typename Lambda> inline void adjacent_nodes(const NodeID node, Lambda &&l) const {
    iterate_neighborhood(node, [&](const EdgeID incident_edge, const NodeID adjacent_node) {
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
  inline void pfor_neighbors(
      const NodeID node, const NodeID max_neighbor_count, const NodeID grainsize, Lambda &&l
  ) const {
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

  [[nodiscard]] inline StaticArray<NodeID> &&take_raw_permutation() {
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

  void update_total_node_weight() final;

  void remove_isolated_nodes(const NodeID isolated_nodes);

  void integrate_isolated_nodes();

  // Compressions statistics

  /*!
   * Returns the number of nodes which have high degree.
   *
   * @returns The number of nodes which have high degree.
   */
  [[nodiscard]] std::size_t high_degree_count() const {
    return _high_degree_count;
  }

  /*!
   * Returns the number of parts that result from splitting the neighborhood of high degree nodes.
   *
   * @returns The number of parts that result from splitting the neighborhood of high degree nodes.
   */
  [[nodiscard]] std::size_t part_count() const {
    return _part_count;
  }

  /*!
   * Returns the number of nodes/parts which use interval encoding.
   *
   * @returns The number of nodes/parts which use interval encoding.
   */
  [[nodiscard]] std::size_t interval_count() const {
    return _interval_count;
  }

  /*!
   * Returns the compression ratio.
   *
   * @return The compression ratio.
   */
  [[nodiscard]] double compression_ratio() const {
    std::size_t uncompressed_size = (n() + 1) * sizeof(EdgeID) + m() * sizeof(NodeID);
    std::size_t compressed_size = _nodes.allocated_size() + _compressed_edges.size();

    if (node_weighted()) {
      uncompressed_size += n() * sizeof(NodeWeight);
      compressed_size += n() * sizeof(NodeWeight);
    }

    if (edge_weighted()) {
      uncompressed_size += m() * sizeof(EdgeWeight);
      compressed_size += m() * sizeof(EdgeWeight);
    }

    return uncompressed_size / (double)compressed_size;
  }

  /**
   * Returns the size reduction in bytes gained by the compression.
   *
   * @returns The size reduction in bytes gained by the compression.
   */
  [[nodiscard]] std::int64_t size_reduction() const {
    std::size_t uncompressed_size = (n() + 1) * sizeof(EdgeID) + m() * sizeof(NodeID);
    std::size_t compressed_size = _nodes.allocated_size() + _compressed_edges.size();

    if (node_weighted()) {
      uncompressed_size += n() * sizeof(NodeWeight);
      compressed_size += n() * sizeof(NodeWeight);
    }

    if (edge_weighted()) {
      uncompressed_size += m() * sizeof(EdgeWeight);
      compressed_size += m() * sizeof(EdgeWeight);
    }

    return uncompressed_size - compressed_size;
  }

  /*!
   * Returns the amount of memory in bytes used by the data structure.
   *
   * @return The amount of memory in bytes used by the data structure.
   */
  [[nodiscard]] std::size_t used_memory() const {
    return _nodes.allocated_size() + _compressed_edges.size() +
           _node_weights.size() * sizeof(NodeWeight) + _edge_weights.size() * sizeof(EdgeWeight);
  }

private:
  CompactStaticArray<EdgeID> _nodes;
  StaticArray<std::uint8_t> _compressed_edges;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  EdgeID _edge_count;
  NodeID _max_degree;

  bool _sorted;

  NodeWeight _total_node_weight = kInvalidNodeWeight;
  EdgeWeight _total_edge_weight = kInvalidEdgeWeight;
  NodeWeight _max_node_weight = kInvalidNodeWeight;

  StaticArray<NodeID> _permutation;

  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;

  std::size_t _high_degree_count;
  std::size_t _part_count;
  std::size_t _interval_count;

  void init_degree_buckets();

  inline std::tuple<EdgeID, NodeID, bool, std::size_t> decode_header(
      const NodeID node, const std::uint8_t *node_data, const std::uint8_t *next_node_data
  ) const {
    const auto [first_edge, next_first_edge, uses_intervals, len] = [&] {
      if constexpr (CompressedGraph::kIntervalEncoding) {
        auto [first_edge, marker_set, len] = marked_varint_decode<EdgeID>(node_data);
        auto [next_first_edge, _, __] = marked_varint_decode<EdgeID>(next_node_data);

        return std::make_tuple(first_edge, next_first_edge, marker_set, len);

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

  template <bool max_edges = false, bool parallel = false, typename Lambda>
  inline void iterate_neighborhood(
      const NodeID node, Lambda &&l, NodeID max_neighbor_count = std::numeric_limits<NodeID>::max()
  ) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return;
    }

    const auto [first_edge, degree, uses_intervals, len] =
        decode_header(node, node_data, next_node_data);
    node_data += len;

    max_neighbor_count = std::min(max_neighbor_count, degree);

    if constexpr (kHighDegreeEncoding) {
      const bool split_neighbourhood = degree >= kHighDegreeThreshold;

      if (split_neighbourhood) {
        iterate_high_degree_neighborhood<max_edges, parallel>(
            node_data, node, first_edge, degree, max_neighbor_count, std::forward<Lambda>(l)
        );
        return;
      }
    }

    const EdgeID max_edge = first_edge + max_neighbor_count;
    invoke_maybe_indirect<std::is_invocable_v<Lambda, EdgeID, NodeID>>(
        std::forward<Lambda>(l),
        [&](auto &&l2) {
          iterate_edges<max_edges>(
              node_data,
              node,
              degree,
              first_edge,
              max_edge,
              uses_intervals,
              std::forward<decltype(l2)>(l2)
          );
        }
    );
  }

  template <bool max_edges, bool parallel, typename Lambda>
  inline void iterate_high_degree_neighborhood(
      const std::uint8_t *data,
      const NodeID node,
      const NodeID first_edge,
      const NodeID degree,
      const NodeID max_neighbor_count,
      Lambda &&l
  ) const {
    const NodeID part_count = math::div_ceil(degree, kHighDegreePartLength);
    const NodeID max_part_count =
        std::min(part_count, math::div_ceil(max_neighbor_count, kHighDegreePartLength));
    const NodeID max_neighbor_rem = ((max_neighbor_count % kHighDegreePartLength) == 0)
                                        ? kHighDegreePartLength
                                        : (max_neighbor_count % kHighDegreePartLength);

    const auto iterate_part = [&](const NodeID part) {
      const std::uint8_t *part_data = data + *((NodeID *)(data + sizeof(NodeID) * part));
      const EdgeID part_first_edge = first_edge + kHighDegreePartLength * part;

      const bool last_part = part + 1 == max_part_count;

      if (last_part) {
        const NodeID part_degree = (part == part_count - 1)
                                       ? (degree - kHighDegreePartLength * (part_count - 1))
                                       : kHighDegreePartLength;
        const EdgeID part_max_edge = part_first_edge + max_neighbor_rem;

        invoke_maybe_indirect<std::is_invocable_v<Lambda, EdgeID, NodeID>>(
            std::forward<Lambda>(l),
            [&](auto &&l2) {
              iterate_edges<max_edges>(
                  part_data,
                  node,
                  part_degree,
                  part_first_edge,
                  part_max_edge,
                  true,
                  std::forward<decltype(l2)>(l2)
              );
            }
        );
      } else {
        const NodeID part_degree = kHighDegreePartLength;
        const EdgeID part_max_edge = part_first_edge + part_degree;

        invoke_maybe_indirect<std::is_invocable_v<Lambda, EdgeID, NodeID>>(
            std::forward<Lambda>(l),
            [&](auto &&l2) {
              iterate_edges<false>(
                  part_data,
                  node,
                  part_degree,
                  part_first_edge,
                  part_max_edge,
                  true,
                  std::forward<decltype(l2)>(l2)
              );
            }
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
  }

  template <bool max_edges, typename Lambda>
  inline void iterate_edges(
      const std::uint8_t *data,
      const NodeID node,
      const NodeID degree,
      const EdgeID first_edge,
      const EdgeID max_edge,
      const bool uses_intervals,
      Lambda &&l
  ) const {
    constexpr bool non_stoppable =
        std::is_void<std::invoke_result_t<Lambda, EdgeID, NodeID>>::value;

    EdgeID edge = first_edge;
    EdgeID gap_edges = degree - 1;

    if constexpr (kIntervalEncoding) {
      if (uses_intervals) {
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

          const NodeID max_interval_len = [&] {
            if constexpr (max_edges) {
              return std::min(cur_interval_len, static_cast<NodeID>(max_edge - edge));
            } else {
              return cur_interval_len;
            }
          }();
          gap_edges -= cur_interval_len;

          for (NodeID j = 0; j < max_interval_len; ++j) {
            if constexpr (non_stoppable) {
              l(edge++, cur_left_extreme + j);
            } else {
              const bool stop = l(edge++, cur_left_extreme + j);
              if (stop) {
                return;
              }
            }
          }
        }
      }
    }

    if (edge == max_edge) {
      return;
    }

    const auto [first_gap, first_gap_len] = signed_varint_decode<SignedID>(data);
    data += first_gap_len;

    const NodeID first_adjacent_node = static_cast<NodeID>(first_gap + node);
    NodeID prev_adjacent_node = first_adjacent_node;

    if constexpr (non_stoppable) {
      l(edge++, first_adjacent_node);
    } else {
      const bool stop = l(edge++, first_adjacent_node);
      if (stop) {
        return;
      }
    }

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
      VarIntRunLengthDecoder<NodeID> rl_decoder(data);
      rl_decoder.decode(max_edge - edge, std::forward<decltype(handle_gap)>(handle_gap));
    } else if constexpr (kStreamEncoding) {
      VarIntStreamDecoder<NodeID> sv_encoder(data, gap_edges);
      sv_encoder.decode(max_edge - edge, std::forward<decltype(handle_gap)>(handle_gap));
    } else {
      while (edge != max_edge) {
        const auto [gap, gap_len] = varint_decode<NodeID>(data);
        data += gap_len;

        const NodeID adjacent_node = gap + prev_adjacent_node + 1;
        prev_adjacent_node = adjacent_node;

        if constexpr (non_stoppable) {
          l(edge++, adjacent_node);
        } else {
          const bool stop = l(edge++, adjacent_node);
          if (stop) {
            return;
          }
        }
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
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;
  using SignedID = CompressedGraph::SignedID;

  /*!
   * Gives an upper limit for the size of the compressed edge array in bytes.
   *
   * @param node_count The number of nodes in the graph.
   * @param edge_count The number of edges in the graph.
   * @return The max size in bytes of the compressed edge array.
   */
  [[nodiscard]] static std::size_t
  compressed_edge_array_max_size(const NodeID node_count, const EdgeID edge_count);

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
   * @param sorted Whether the nodes to add are stored by deg-buckets order.
   */
  void init(
      const NodeID node_count,
      const EdgeID edge_count,
      const bool store_node_weights,
      const bool store_edge_weights,
      const bool sorted
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
   * Returns the used memory of the compressed edge array.
   *
   * @return The used memory of the compressed edge array.
   */
  std::size_t edge_array_size() const;

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
  CompactStaticArray<EdgeID> _nodes;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  bool _store_node_weights;
  bool _store_edge_weights;
  std::int64_t _total_node_weight;
  std::int64_t _total_edge_weight;

  bool _sorted;

  std::uint8_t *_compressed_edges;
  std::uint8_t *_cur_compressed_edges;

  EdgeID _edge_count;
  NodeID _max_degree;

  bool _first_isolated_node;
  EdgeID _last_real_edge;

  std::size_t _high_degree_count;
  std::size_t _part_count;
  std::size_t _interval_count;

  void add_edges(
      const NodeID node,
      std::uint8_t *marked_byte,
      std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
  );
};

} // namespace kaminpar::shm
