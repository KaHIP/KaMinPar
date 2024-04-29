/*******************************************************************************
 * Compressed static graph representation.
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
   * @param num_high_degree_nodes The number of nodes that have high degree.
   * @param num_high_degree_parts The total number of parts that result from splitting high degree
   * neighborhoods.
   * @param num_interval_nodes The number of nodes that have at least one interval in its
   * neighborhood.
   * @param num_intervals The total number of intervals.
   */
  explicit CompressedGraph(
      CompactStaticArray<EdgeID> nodes,
      StaticArray<std::uint8_t> compressed_edges,
      StaticArray<NodeWeight> node_weights,
      StaticArray<EdgeWeight> edge_weights,
      EdgeID edge_count,
      NodeID max_degree,
      bool sorted,
      std::size_t num_high_degree_nodes,
      std::size_t num_high_degree_parts,
      std::size_t num_interval_nodes,
      std::size_t num_intervals
  );

  CompressedGraph(const CompressedGraph &) = delete;
  CompressedGraph &operator=(const CompressedGraph &) = delete;

  CompressedGraph(CompressedGraph &&) noexcept = default;
  CompressedGraph &operator=(CompressedGraph &&) noexcept = default;

  template <typename Lambda> decltype(auto) reified(Lambda &&l) const {
    return l(*this);
  }

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
  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID node) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

    const bool is_isolated_node = node_data == next_node_data;
    if (is_isolated_node) {
      return {0, 0};
    }

    const auto [first_edge, degree, _, __] = decode_header(node, node_data, next_node_data);
    return {first_edge, first_edge + degree};
  }

  template <typename Lambda> void adjacent_nodes(const NodeID node, Lambda &&l) const {
    decode_neighborhood(node, [&](const EdgeID incident_edge, const NodeID adjacent_node) {
      l(adjacent_node);
    });
  }

  template <typename Lambda> void neighbors(const NodeID node, Lambda &&l) const {
    decode_neighborhood(node, std::forward<Lambda>(l));
  }

  template <typename Lambda>
  void neighbors(const NodeID node, const NodeID max_neighbor_count, Lambda &&l) const {
    decode_neighborhood(node, std::forward<Lambda>(l));
  }

  template <typename Lambda>
  void pfor_neighbors(
      const NodeID node, const NodeID max_neighbor_count, const NodeID grainsize, Lambda &&l
  ) const {
    decode_neighborhood<true>(node, std::forward<Lambda>(l));
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

  void update_total_node_weight() final;

  void remove_isolated_nodes(const NodeID isolated_nodes);

  void integrate_isolated_nodes();

  // Compressions statistics

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

    return uncompressed_size / static_cast<double>(compressed_size);
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

  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;

  void init_degree_buckets();

  inline std::tuple<EdgeID, NodeID, bool, std::size_t> decode_header(
      const NodeID node, const std::uint8_t *node_data, const std::uint8_t *next_node_data
  ) const {
    const auto [first_edge, next_first_edge, uses_intervals, len] = [&] {
      if constexpr (CompressedGraph::kIntervalEncoding) {
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

  template <bool parallel = false, typename Lambda>
  void decode_neighborhood(const NodeID node, Lambda &&l) const {
    const std::uint8_t *data = _compressed_edges.data();

    const std::uint8_t *node_data = data + _nodes[node];
    const std::uint8_t *next_node_data = data + _nodes[node + 1];

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
        decode_parts<parallel>(node_data, node, edge, degree, std::forward<Lambda>(l));
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

/*!
 * A builder that constructs compressed graphs in a single read pass. It does this by
 * overcommiting memory for the compressed edge array.
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
   * @param sorted Whether the nodes to add are stored in degree-bucket order.
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
   * @param neighbourhood The neighbourhood of the node to add, which consits of the adjacent
   * nodes and the corresponding edge weights.
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
  [[nodiscard]] std::size_t edge_array_size() const;

  /*!
   * Returns the total weight of the nodes that have been added.
   *
   * @return The total weight of the nodes that have been added.
   */
  [[nodiscard]] std::int64_t total_node_weight() const;

  /*!
   * Returns the total weight of the edges that have been added.
   *
   * @return The total weight of the edges that have been added.
   */
  [[nodiscard]] std::int64_t total_edge_weight() const;

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
  EdgeID _effective_last_edge_offset;

  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;

  void add_edges(
      const NodeID node,
      std::uint8_t *marked_byte,
      std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
  );
};

} // namespace kaminpar::shm
