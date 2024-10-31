/*******************************************************************************
 * Static compressed graph representation.
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

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {

/*!
 * A compressed static graph that stores the nodes and edges in a compressed adjacency array. It
 * uses variable length encoding, gap encoding and interval encoding to compress the edge array.
 * Additionally, it stores the edge weights interleaved with the edges and stores them with variable
 * length encoding and gap encoding.
 */
class CompressedGraph : public AbstractGraph {
  using CompressedNeighborhoods = kaminpar::CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight>;

public:
  using AbstractGraph::EdgeID;
  using AbstractGraph::EdgeWeight;
  using AbstractGraph::NodeID;
  using AbstractGraph::NodeWeight;

  /*!
   * Whether edge weights are compressed.
   */
  static constexpr bool kCompressEdgeWeights = CompressedNeighborhoods::kCompressEdgeWeights;

  /*!
   * Whether high degree encoding is used.
   */
  static constexpr bool kHighDegreeEncoding = CompressedNeighborhoods::kHighDegreeEncoding;

  /*!
   * The minimum degree of a node to be considered high degree.
   */
  static constexpr NodeID kHighDegreeThreshold = CompressedNeighborhoods::kHighDegreeThreshold;

  /*!
   * The length of a part when splitting the neighbourhood of a high degree
   * node.
   */
  static constexpr NodeID kHighDegreePartLength = CompressedNeighborhoods::kHighDegreePartLength;

  /*!
   * Whether interval encoding is used.
   */
  static constexpr bool kIntervalEncoding = CompressedNeighborhoods::kIntervalEncoding;

  /*!
   * The minimum length of an interval to encode if interval encoding is used.
   */
  static constexpr NodeID kIntervalLengthTreshold =
      CompressedNeighborhoods::kIntervalLengthTreshold;

  /*!
   * Whether run-length encoding is used.
   */
  static constexpr bool kRunLengthEncoding = CompressedNeighborhoods::kRunLengthEncoding;

  /*!
   * Whether StreamVByte encoding is used.
   */
  static constexpr bool kStreamVByteEncoding = CompressedNeighborhoods::kStreamVByteEncoding;

  /*!
   * Constructs a new compressed graph.
   *
   * @param compressed_neighborhoods The nodes, edges and edge weights that are stored in compressed
   * form.
   * @param node_weights The node weights.
   * @param sorted Whether the nodes are stored in degree-buckets order.
   */
  explicit CompressedGraph(
      CompressedNeighborhoods compressed_neighborhoods,
      StaticArray<NodeWeight> node_weights,
      bool sorted
  );

  CompressedGraph(const CompressedGraph &) = delete;
  CompressedGraph &operator=(const CompressedGraph &) = delete;

  CompressedGraph(CompressedGraph &&) noexcept = default;
  CompressedGraph &operator=(CompressedGraph &&) noexcept = default;

  //
  // Size of the graph
  //

  [[nodiscard]] NodeID n() const final {
    return _compressed_neighborhoods.num_nodes();
  }

  [[nodiscard]] EdgeID m() const final {
    return _compressed_neighborhoods.num_edges();
  }

  //
  // Node and edge weights
  //

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

  void update_total_node_weight() final;

  [[nodiscard]] inline bool is_edge_weighted() const final {
    return _compressed_neighborhoods.has_edge_weights();
  }

  [[nodiscard]] inline EdgeWeight total_edge_weight() const final {
    return _compressed_neighborhoods.total_edge_weight();
  }

  //
  // Iterators for nodes / edges
  //

  [[nodiscard]] IotaRange<NodeID> nodes() const final {
    return {static_cast<NodeID>(0), n()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> edges() const final {
    return {static_cast<EdgeID>(0), m()};
  }

  [[nodiscard]] inline IotaRange<EdgeID> incident_edges(const NodeID node) const final {
    return _compressed_neighborhoods.incident_edges(node);
  }

  //
  // Node degree
  //

  [[nodiscard]] inline NodeID max_degree() const final {
    return _compressed_neighborhoods.max_degree();
  }

  [[nodiscard]] inline NodeID degree(const NodeID node) const final {
    return _compressed_neighborhoods.degree(node);
  }

  //
  // Graph operations
  //

  template <typename Lambda> inline void adjacent_nodes(const NodeID u, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    _compressed_neighborhoods.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if constexpr (kDecodeEdgeWeights) {
        return l(v, w);
      } else {
        return l(v);
      }
    });
  }

  template <typename Lambda> inline void neighbors(const NodeID u, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    _compressed_neighborhoods.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if constexpr (kDecodeEdgeWeights) {
        return l(e, v, w);
      } else {
        return l(e, v);
      }
    });
  }

  template <typename Lambda>
  inline void neighbors(const NodeID u, const NodeID max_num_neighbors, Lambda &&l) const {
    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, EdgeID, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    _compressed_neighborhoods
        .neighbors(u, max_num_neighbors, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
          if constexpr (kDecodeEdgeWeights) {
            return l(e, v, w);
          } else {
            return l(e, v);
          }
        });
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_nodes(Lambda &&l) const {
    tbb::parallel_for(static_cast<NodeID>(0), n(), std::forward<Lambda>(l));
  }

  template <typename Lambda> inline void pfor_edges(Lambda &&l) const {
    tbb::parallel_for(static_cast<EdgeID>(0), m(), std::forward<Lambda>(l));
  }

  template <typename Lambda>
  inline void pfor_neighbors(
      const NodeID u,
      [[maybe_unused]] const NodeID max_num_neighbors,
      [[maybe_unused]] const NodeID grainsize,
      Lambda &&l
  ) const {
    // The compressed graph does not allow for arbitrary grainsize. It is also not supported
    // to only visit a subrange of neighbors.
    _compressed_neighborhoods.parallel_neighbors(u, std::forward<Lambda>(l));
  }

  //
  // Graph permutation
  //

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

  //
  // Degree buckets
  //

  [[nodiscard]] inline bool sorted() const final {
    return _sorted;
  }

  [[nodiscard]] inline std::size_t number_of_buckets() const final {
    return _number_of_buckets;
  }

  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const final {
    return _buckets[bucket + 1] - _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const final {
    return _buckets[bucket];
  }

  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const final {
    return first_node_in_bucket(bucket + 1);
  }

  //
  // Isolated nodes
  //

  void remove_isolated_nodes(const NodeID num_isolated_nodes);

  void integrate_isolated_nodes();

  //
  // Compressions statistics
  //

  /*!
   * Returns the number of nodes that have high degree.
   *
   * @returns The number of nodes that have high degree.
   */
  [[nodiscard]] std::size_t num_high_degree_nodes() const {
    return _compressed_neighborhoods.num_high_degree_nodes();
  }

  /*!
   * Returns the total number of parts that result from splitting high degree neighborhoods.
   *
   * @returns The total number of parts that result from splitting high degree neighborhoods.
   */
  [[nodiscard]] std::size_t num_high_degree_parts() const {
    return _compressed_neighborhoods.num_high_degree_parts();
  }

  /*!
   * Returns the number of nodes that have at least one interval.
   *
   * @returns The number of nodes that have at least one interval.
   */
  [[nodiscard]] std::size_t num_interval_nodes() const {
    return _compressed_neighborhoods.num_interval_nodes();
  }

  /*!
   * Returns the total number of intervals.
   *
   * @returns The total number of intervals.
   */
  [[nodiscard]] std::size_t num_intervals() const {
    return _compressed_neighborhoods.num_intervals();
  }

  /*!
   * Returns the compression ratio.
   *
   * @return The compression ratio.
   */
  [[nodiscard]] double compression_ratio() const {
    std::size_t uncompressed_size = (n() + 1) * sizeof(EdgeID) + m() * sizeof(NodeID);
    std::size_t compressed_size = _compressed_neighborhoods.memory_space();

    if (is_node_weighted()) {
      uncompressed_size += n() * sizeof(NodeWeight);
      compressed_size += n() * sizeof(NodeWeight);
    }

    if (is_edge_weighted()) {
      uncompressed_size += m() * sizeof(EdgeWeight);
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
    std::size_t compressed_size = _compressed_neighborhoods.memory_space();

    if (is_node_weighted()) {
      uncompressed_size += n() * sizeof(NodeWeight);
      compressed_size += n() * sizeof(NodeWeight);
    }

    if (is_edge_weighted()) {
      uncompressed_size += m() * sizeof(EdgeWeight);
    }

    return uncompressed_size - compressed_size;
  }

  /*!
   * Returns the amount of memory in bytes used by the data structure.
   *
   * @return The amount of memory in bytes used by the data structure.
   */
  [[nodiscard]] std::size_t used_memory() const {
    return _compressed_neighborhoods.memory_space() + _node_weights.size() * sizeof(NodeWeight);
  }

  //
  // Direct member access -- used for some "low level" operations
  //

  [[nodiscard]] inline CompactStaticArray<EdgeID> &raw_nodes() {
    return _compressed_neighborhoods.raw_nodes();
  }

  [[nodiscard]] inline const CompactStaticArray<EdgeID> &raw_nodes() const {
    return _compressed_neighborhoods.raw_nodes();
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &raw_node_weights() {
    return _node_weights;
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_node_weights() const {
    return _node_weights;
  }

  [[nodiscard]] inline CompactStaticArray<EdgeID> &&take_raw_nodes() {
    return _compressed_neighborhoods.take_raw_nodes();
  }

  [[nodiscard]] inline StaticArray<NodeWeight> &&take_raw_node_weights() {
    return std::move(_node_weights);
  }

  [[nodiscard]] const StaticArray<std::uint8_t> &raw_compressed_edges() const {
    return _compressed_neighborhoods.raw_compressed_edges();
  }

  [[nodiscard]] inline const StaticArray<NodeWeight> &raw_edge_weights() const {
    return _compressed_neighborhoods.raw_edge_weights();
  }

private:
  CompressedNeighborhoods _compressed_neighborhoods;
  StaticArray<NodeWeight> _node_weights;

  NodeWeight _max_node_weight = kInvalidNodeWeight;
  NodeWeight _total_node_weight = kInvalidNodeWeight;

  StaticArray<NodeID> _permutation;
  bool _sorted;
  std::vector<NodeID> _buckets = std::vector<NodeID>(kNumberOfDegreeBuckets<NodeID> + 1);
  std::size_t _number_of_buckets = 0;

  void init_degree_buckets();
};

} // namespace kaminpar::shm
