/*! @file
 * Data structures for static graphs, partitions and graph hierarchies, as well as utility functions.
 */
#pragma once

#include "datastructure/static_array.h"
#include "definitions.h"
#include "parallel.h"
#include "utility/utility.h"

#include "gtest/gtest.h"
#include <numeric>
#include <ranges>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>
#include <utility>
#include <vector>

namespace kaminpar {
class Graph;
class PartitionedGraph;

static constexpr std::size_t kNumberOfDegreeBuckets = std::numeric_limits<NodeID>::digits + 1;

/*!
 * Returns the lowest degree that could occur in a given bucket id. This does not imply that the bucket actually has a
 * node with this degree.
 *
 * @param bucket A bucket id.
 * @return Lowest possible degree of a node placed in the given bucket.
 */
Degree lowest_degree_in_bucket(std::size_t bucket);

/*!
 * Returns the bucket id of a node of given degree. Note: `lowest_degree_in_bucket(degree_bucket(degree)) <= degree`
 * with equality iff. degree is a power of 2 or zero.
 *
 * @param degree Degree of a node.
 * @return ID of the bucket that the node should be placed in.
 */
Degree degree_bucket(Degree degree);

/*!
 * Static weighted graph represented by an adjacency array.
 *
 * Common usage patterns are as follows:
 * ```
 * Graph graph(...);
 *
 * // iterate over all nodes in the graph
 * for (const NodeID u : graph.nodes()) {
 *     // iterate over incident edges *with* their endpoints
 *     for (const auto [e, v] : graph.neighbors(u)) {
 *         // edge e = (u, v)
 *     }
 *
 *     // iterate over adjacent nodes
 *     for (const NodeID v : graph.adjacent_nodes(u)) {
 *         // there is an edge (u, v)
 *     }
 *
 *     // iterate over incident edges *without* their endpoints
 *     for (const EdgeID e : graph.incident_edges(u)) {
 *         // edge e = (u, graph.edge_target(e))
 *     }
 * }
 *
 * // iterate over all edges in the graph *without* their endpoints
 * for (const EdgeID e : graph.edges()) {
 *     // ...
 * }
 * ```
 *
 * Note that this class does not contain a graph partition. To extend the graph with a partition, wrap an object of this
 * class in a kaminpar::PartitionedGraph.
 */
class Graph {
public:
  Graph() {}

  Graph(StaticArray<EdgeID> nodes, StaticArray<NodeID> edges, StaticArray<NodeWeight> node_weights = {},
        StaticArray<EdgeWeight> edge_weights = {}, bool sorted = false);

  Graph(tag::Sequential, StaticArray<EdgeID> nodes, StaticArray<NodeID> edges,
        StaticArray<NodeWeight> node_weights = {}, StaticArray<EdgeWeight> edge_weights = {}, bool sorted = false);

  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;

  // clang-format off
  [[nodiscard]] inline auto &raw_nodes() { return _nodes; }
  [[nodiscard]] inline const auto &raw_nodes() const { return _nodes; }
  [[nodiscard]] inline const auto &raw_edges() const { return _edges; }
  [[nodiscard]] inline auto &raw_node_weights() { return _node_weights; }
  [[nodiscard]] inline const auto &raw_node_weights() const { return _node_weights; }
  [[nodiscard]] inline const auto &raw_edge_weights() const { return _edge_weights; }
  [[nodiscard]] inline auto &&take_raw_nodes() { return std::move(_nodes); }
  [[nodiscard]] inline auto &&take_raw_edges() { return std::move(_edges); }
  [[nodiscard]] inline auto &&take_raw_node_weights() { return std::move(_node_weights); }
  [[nodiscard]] inline auto &&take_raw_edge_weights() { return std::move(_edge_weights); }

  // Edge and node weights
  [[nodiscard]] inline NodeWeight total_node_weight() const { return _total_node_weight; }
  [[nodiscard]] inline EdgeWeight total_edge_weight() const { return _total_edge_weight; }
  [[nodiscard]] inline const StaticArray<NodeWeight> &node_weights() const { return _node_weights; }
  [[nodiscard]] inline bool is_node_weighted() const { return static_cast<NodeWeight>(n()) != total_node_weight(); }
  [[nodiscard]] inline bool is_edge_weighted() const { return static_cast<EdgeWeight>(m()) != total_edge_weight(); }
  [[nodiscard]] inline NodeID n() const { return static_cast<NodeID>(_nodes.size() - 1); }
  [[nodiscard]] inline NodeID last_node() const { return n() - 1; }
  [[nodiscard]] inline EdgeID m() const { return static_cast<EdgeID>(_edges.size()); }
  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const { return is_node_weighted() ? _node_weights[u] : 1; }
  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const { return is_edge_weighted() ? _edge_weights[e] : 1; }
  [[nodiscard]] inline NodeWeight max_node_weight() const { return _max_node_weight; }

  // Graph structure
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const { return _edges[e]; }
  [[nodiscard]] inline NodeID degree(const NodeID u) const { return static_cast<NodeID>(_nodes[u + 1] - _nodes[u]); }
  [[nodiscard]] inline EdgeID first_edge(const NodeID u) const { return _nodes[u]; }
  [[nodiscard]] inline EdgeID first_invalid_edge(const NodeID u) const { return _nodes[u + 1]; }

  // Iterators for nodes / edges
  [[nodiscard]] inline auto nodes() const { return std::views::iota(static_cast<NodeID>(0), n()); }
  [[nodiscard]] inline auto edges() const { return std::views::iota(static_cast<EdgeID>(0), m()); }
  [[nodiscard]] inline auto incident_edges(const NodeID u) const { return std::views::iota(_nodes[u], _nodes[u + 1]); }
  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const { return std::views::iota(_nodes[u], _nodes[u + 1]) | std::views::transform([this](const EdgeID e) { return this->edge_target(e); }); }
  [[nodiscard]] inline auto neighbors(const NodeID u) const { return std::views::iota(_nodes[u], _nodes[u + 1]) | std::views::transform([this](const EdgeID e) { return std::make_pair(e, this->edge_target(e)); }); }

  // Degree buckets
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const { return _buckets[bucket + 1] - _buckets[bucket]; }
  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const { return _buckets[bucket]; }
  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const { return first_node_in_bucket(bucket + 1); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return _number_of_buckets; }
  [[nodiscard]] inline bool sorted() const { return _sorted; }
  // clang-format on

#ifdef KAMINPAR_ENABLE_DEBUG_FEATURES
  [[nodiscard]] inline const std::string &name() const { return _name; }
  inline void set_name(const std::string &name) { _name = name; }
#endif // KAMINPAR_ENABLE_DEBUG_FEATURES

  void print() const;
  void print_weighted() const;

  void update_total_node_weight();

private:
  FRIEND_TEST(AWeightedGridGraph, ContractingToSingletonClustersWorks);
  FRIEND_TEST(AWeightedGridGraph, ContractingGridHorizontallyWorks);
  FRIEND_TEST(AWeightedGridGraph, ContractingGridVerticallyWorks);

  void init_degree_buckets();

  StaticArray<EdgeID> _nodes;
  StaticArray<NodeID> _edges;
  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;
  NodeWeight _total_node_weight;
  EdgeWeight _total_edge_weight;
  NodeWeight _max_node_weight{1};
  bool _sorted;
  std::vector<NodeID> _buckets{std::vector<NodeID>(kNumberOfDegreeBuckets + 1)};
  std::size_t _number_of_buckets{0};
#ifdef KAMINPAR_ENABLE_DEBUG_FEATURES
  std::string _name{};
#endif // KAMINPAR_ENABLE_DEBUG_FEATURES
};

class ParallelBalancer;

/*!
 * Extends a kaminpar::Graph with a graph partition.
 *
 * This class implements the same member functions as kaminpar::Graph plus some more that only concern the graph
 * partition. Functions that are also implemented in kaminpar::Graph are delegated to the wrapped object.
 *
 * If an object of this class is constructed without partition, all nodes are marked unassigned, i.e., are placed in
 * block `kInvalidBlockID`.
 */
class PartitionedGraph {
  friend ParallelBalancer;

  static constexpr auto kDebug = false;
  static constexpr NodeID kNodeToDebug = 1;
  friend void copy_subgraph_partitions(PartitionedGraph &p_graph, const scalable_vector<PartitionedGraph> &p_subgraphs,
                                       const scalable_vector<NodeID> &mapping);
  friend void copy_subgraph_partitions(PartitionedGraph &p_graph,
                                       const scalable_vector<StaticArray<BlockID>> &p_subgraph_partitions,
                                       const BlockID k_per_subgraph, const BlockID final_k_per_subgraph,
                                       const scalable_vector<NodeID> &mapping);

public:
  PartitionedGraph(const Graph &graph, BlockID k, StaticArray<BlockID> partition = {},
                   scalable_vector<BlockID> final_k = {});
  PartitionedGraph(tag::Sequential, const Graph &graph, BlockID k, StaticArray<BlockID> partition = {},
                   scalable_vector<BlockID> final_k = {});
  PartitionedGraph() : _graph{nullptr} {}

  PartitionedGraph(const PartitionedGraph &) = delete;
  PartitionedGraph &operator=(const PartitionedGraph &) = delete;
  PartitionedGraph(PartitionedGraph &&) noexcept = default;
  PartitionedGraph &operator=(PartitionedGraph &&other) noexcept = default;

  [[nodiscard]] inline bool initialized() const { return _graph != nullptr; }

  //
  // Delegates to _graph
  //

  // clang-format off
  [[nodiscard]] inline const Graph &graph() const { return *_graph; }

  [[nodiscard]] inline NodeID n() const { return _graph->n(); }
  [[nodiscard]] inline NodeID last_node() const { return n() - 1; }
  [[nodiscard]] inline EdgeID m() const { return _graph->m(); }
  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const { return _graph->node_weight(u); }
  [[nodiscard]] inline EdgeWeight edge_weight(const EdgeID e) const { return _graph->edge_weight(e); }
  [[nodiscard]] inline NodeWeight total_node_weight() const { return _graph->total_node_weight(); }
  [[nodiscard]] inline NodeWeight max_node_weight() const { return _graph->max_node_weight(); }
  [[nodiscard]] inline EdgeWeight total_edge_weight() const { return _graph->total_edge_weight(); }
  [[nodiscard]] inline NodeID edge_target(const EdgeID e) const { return _graph->edge_target(e); }
  [[nodiscard]] inline NodeID degree(const NodeID u) const { return _graph->degree(u); }
  [[nodiscard]] inline auto nodes() const { return _graph->nodes(); }
  [[nodiscard]] inline auto edges() const { return _graph->edges(); }
  [[nodiscard]] inline auto adjacent_nodes(const NodeID u) const { return _graph->adjacent_nodes(u); }
  [[nodiscard]] inline auto neighbors(const NodeID u) const { return _graph->neighbors(u); }
  [[nodiscard]] inline auto incident_edges(const NodeID u) const { return _graph->incident_edges(u); }
  [[nodiscard]] inline std::size_t bucket_size(const std::size_t bucket) const { return _graph->bucket_size(bucket); }
  [[nodiscard]] inline NodeID first_node_in_bucket(const std::size_t bucket) const { return _graph->first_node_in_bucket(bucket); }
  [[nodiscard]] inline NodeID first_invalid_node_in_bucket(const std::size_t bucket) const { return _graph->first_invalid_node_in_bucket(bucket); }
  [[nodiscard]] inline std::size_t number_of_buckets() const { return _graph->number_of_buckets(); }
  [[nodiscard]] inline bool sorted() const { return _graph->sorted(); }
  // clang-format on

  //
  // Partition related members
  //

  [[nodiscard]] inline auto blocks() const { return std::views::iota(static_cast<BlockID>(0), k()); }
  [[nodiscard]] inline BlockID block(const NodeID u) const { return _partition[u]; }

  template<bool update_block_weight = true>
  void set_block(const NodeID u, const BlockID new_b) {
    ASSERT(u < n()) << "invalid node id " << u;
    ASSERT(new_b < k()) << "invalid block id " << new_b << " for node " << u;
    ASSERT(new_b != block(u)) << V(u) << V(new_b) << V(block(u));
    DBG << "set_block(" << u << ", " << new_b << ")";

    if constexpr (update_block_weight) {
      if (block(u) != kInvalidBlockID) { _block_weights[block(u)] -= node_weight(u); }
      _block_weights[new_b] += node_weight(u);
    }

    // change block
    _partition[u] = new_b;
  }

  // clang-format off
  [[nodiscard]] inline NodeWeight block_weight(const BlockID b) const { return _block_weights[b]; }
  [[nodiscard]] inline const auto &block_weights() const { return _block_weights; }
  [[nodiscard]] inline auto &&take_block_weights() { return std::move(_block_weights); }
  [[nodiscard]] inline BlockID heaviest_block() const { return std::max_element(_block_weights.begin(), _block_weights.end()) - _block_weights.begin(); }
  [[nodiscard]] inline BlockID lightest_block() const { return std::min_element(_block_weights.begin(), _block_weights.end()) - _block_weights.begin(); }
  [[nodiscard]] inline BlockID k() const { return _k; }
  [[nodiscard]] inline const StaticArray<BlockID> &partition() const { return _partition; }
  [[nodiscard]] inline StaticArray<BlockID> &&take_partition() { return std::move(_partition); }
  // clang-format on

  void change_k(BlockID new_k);

  [[nodiscard]] inline BlockID final_k(const BlockID b) const { return _final_k[b]; }
  [[nodiscard]] inline const scalable_vector<BlockID> &final_ks() const { return _final_k; }
  [[nodiscard]] inline scalable_vector<BlockID> &&take_final_k() { return std::move(_final_k); }
  inline void set_final_k(const BlockID b, const BlockID final_k) { _final_k[b] = final_k; }
  inline void set_final_ks(scalable_vector<BlockID> final_ks) { _final_k = std::move(final_ks); }

  //
  // Block / graph names
  //

#ifdef KAMINPAR_ENABLE_DEBUG_FEATURES
  [[nodiscard]] inline const std::string &name() const { return _graph->name(); }
  inline void set_name(const std::string &name) { _graph->set_name(name); }
  [[nodiscard]] inline const std::vector<std::string> &block_names() const { return _block_names; }
  inline void set_block_names(const std::vector<std::string> &block_names) { _block_names = block_names; }
  [[nodiscard]] inline const std::string &block_name(const BlockID b) const { return _block_names[b]; }
  inline void set_block_name(const BlockID b, const std::string &name) { _block_names[b] = name; }
#endif // KAMINPAR_ENABLE_DEBUG_FEATURES

private:
  void init_block_weights() {
    tbb::enumerable_thread_specific<std::vector<BlockWeight>> tl_block_weights{
        [&] { return std::vector<BlockWeight>(k()); }};
    tbb::parallel_for(tbb::blocked_range(static_cast<NodeID>(0), n()), [&](auto &r) {
      auto &local_block_weights = tl_block_weights.local();
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        if (block(u) != kInvalidBlockID) { local_block_weights[block(u)] += node_weight(u); }
      }
    });

    tbb::parallel_for(static_cast<BlockID>(0), k(), [&](const BlockID b) {
      BlockWeight sum = 0;
      for (auto &local_block_weights : tl_block_weights) { sum += local_block_weights[b]; }
      _block_weights[b] = sum;
    });
  }

  void init_block_weights_seq() {
    for (const NodeID u : nodes()) {
      if (block(u) != kInvalidBlockID) { _block_weights[block(u)] += node_weight(u); }
    }
  }

  void reinit_block_weights() {
    for (const BlockID b : blocks()) { _block_weights[b] = 0; }
    init_block_weights();
  }

  //! This is the underlying graph structure.
  const Graph *_graph;
  //! Number of blocks in this partition.
  BlockID _k;
  //! The partition, holds the block id [0, k) for each node.
  StaticArray<BlockID> _partition; // O(n)
  //! Current weight of each block.
  StaticArray<parallel::IntegralAtomicWrapper<NodeWeight>> _block_weights; // O(n)
  //! For each block in the current partition, this is the number of blocks that we want to split the block in the
  //! final partition. For instance, after the first bisection, this might be {_k / 2, _k / 2}, although other values
  //! are possible when using adaptive k's or if _k is not a power of 2.
  scalable_vector<BlockID> _final_k; // O(k)

#ifdef KAMINPAR_ENABLE_DEBUG_FEATURES
  std::vector<std::string> _block_names{};
#endif // KAMINPAR_ENABLE_DEBUG_FEATURES
};
} // namespace kaminpar