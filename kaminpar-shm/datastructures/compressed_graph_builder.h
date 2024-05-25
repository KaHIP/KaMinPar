/*******************************************************************************
 * Sequential and parallel builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/compressed_graph.h"

namespace kaminpar::shm {

class CompressedEdgesBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;
  using SignedID = CompressedGraph::SignedID;

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
  );

  CompressedEdgesBuilder(const CompressedEdgesBuilder &) = delete;
  CompressedEdgesBuilder &operator=(const CompressedEdgesBuilder &) = delete;

  CompressedEdgesBuilder(CompressedEdgesBuilder &&) noexcept = default;

  /*!
   * Initializes/resets the builder.
   *
   * @param first_edge The first edge ID of the first node to be added.
   */
  void init(const EdgeID first_edge);

  /*!
   * Adds the neighborhood of a node. Note that the neighbourhood vector is modified.
   *
   * @param node The node whose neighborhood to add.
   * @param neighbourhood The neighbourhood of the node to add.
   * @return The offset into the compressed edge array of the node.
   */
  EdgeID add(const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood);

  /*!
   * Returns the number of bytes that the compressed data of the added neighborhoods take up.
   *
   * @return The number of bytes that the compressed data of the added neighborhoods take up.
   */
  [[nodiscard]] std::size_t size() const;

  /*!
   * Returns a pointer to the start of the compressed data.
   *
   * @return A pointer to the start of the compressed data.
   */
  [[nodiscard]] const std::uint8_t *compressed_data() const;

  /*!
   * Returns ownership of the compressed data
   *
   * @return Ownership of the compressed data.
   */
  [[nodiscard]] heap_profiler::unique_ptr<std::uint8_t> take_compressed_data();

  [[nodiscard]] std::size_t max_degree() const;
  [[nodiscard]] std::int64_t total_edge_weight() const;

  [[nodiscard]] std::size_t num_high_degree_nodes() const;
  [[nodiscard]] std::size_t num_high_degree_parts() const;
  [[nodiscard]] std::size_t num_interval_nodes() const;
  [[nodiscard]] std::size_t num_intervals() const;

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

  template <typename Container>
  void add_edges(const NodeID node, std::uint8_t *marked_byte, Container &&neighbourhood);
};

/*!
 * A sequential builder that constructs compressed graphs.
 */
class CompressedGraphBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;
  using SignedID = CompressedGraph::SignedID;

public:
  /*!
   * Compresses a graph in compressed sparse row format.
   *
   * @param graph The graph to compress.
   * @return The compressed input graph.
   */
  static CompressedGraph compress(const CSRGraph &graph);

  /*!
   * Constructs a new CompressedGraphBuilder.
   *
   * @param node_count The number of nodes of the graph to compress.
   * @param edge_count The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes to add are stored in degree-bucket order.
   */
  CompressedGraphBuilder(
      const NodeID node_count,
      const EdgeID edge_count,
      const bool has_node_weights,
      const bool has_edge_weights,
      const bool sorted
  );

  /*!
   * Adds a node to the compressed graph. Note that the neighbourhood vector is modified.
   *
   * @param node The node to add.
   * @param neighbourhood The neighbourhood of the node to add.
   */
  void add_node(const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood);

  /*!
   * Adds a node weight to the compressed graph.
   *
   * @param node The node whose weight to add.
   * @param weight The weight to store.
   */
  void add_node_weight(const NodeID node, const NodeWeight weight);

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
  [[nodiscard]] std::size_t currently_used_memory() const;

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
  // The arrays that store information about the compressed graph
  CompactStaticArray<EdgeID> _nodes;
  bool _sorted; // Whether the nodes of the graph are stored in degree-bucket order

  CompressedEdgesBuilder _compressed_edges_builder;
  EdgeID _num_edges;

  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  // Statistics about the graph
  bool _store_node_weights;
  std::int64_t _total_node_weight;
};

class ParallelCompressedGraphBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;

public:
  /*!
   * Compresses a graph stored in compressed sparse row format.
   *
   * @param graph The graph to compress.
   * @return The compressed graph.
   */
  [[nodiscard]] static CompressedGraph compress(const CSRGraph &graph);

  /*!
   * Initializes the builder by allocating memory for the various arrays.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes to add are stored in degree-bucket order.
   */
  ParallelCompressedGraphBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
      const bool has_node_weights,
      const bool has_edge_weights,
      const bool sorted
  );

  /*!
   * Adds a node to the compressed graph.
   *
   * @param node The node to add.
   * @param offset The offset into the compressed edge array at which the compressed neighborhood of
   * the node is stored.
   */
  void add_node(const NodeID node, const EdgeID offset);

  /**
   * Adds compressed neighborhoods of possible multiple consecutive nodes to the compressed graph.
   *
   * @param offset The offset into the compressed edge array at which the compressed neighborhoods
   * are stored.
   * @param length The length in bytes of the compressed neighborhoods to store.
   * @param data A pointer to the start of the compressed neighborhoods to copy.
   */
  void add_compressed_edges(const EdgeID offset, const EdgeID length, const std::uint8_t *data);

  /*!
   * Adds a node weight to the compressed graph.
   *
   * @param node The node whose weight to add.
   * @param weight The weight to store.
   */
  void add_node_weight(const NodeID node, const NodeWeight weight);

  /*!
   * Returns a reference to the edge weights of the compressed graph.
   *
   * @return A reference to the edge weights of the compressed graph.
   */
  [[nodiscard]] StaticArray<EdgeWeight> &edge_weights();

  /*!
   * Adds (cummulative) statistics about nodes of the compressed graph.
   */
  void record_local_statistics(
      NodeID max_degree,
      NodeWeight node_weight,
      EdgeWeight edge_weight,
      std::size_t num_high_degree_nodes,
      std::size_t num_high_degree_parts,
      std::size_t num_interval_nodes,
      std::size_t num_intervals
  );

  /*!
   * Finalizes the compressed graph. Note that all nodes, compressed neighborhoods, node weights and
   * edge weights have to be added at this point.
   *
   * @return The resulting compressed graph.
   */
  [[nodiscard]] CompressedGraph build();

private:
  // The arrays that store information about the compressed graph
  CompactStaticArray<EdgeID> _nodes;
  bool _sorted; // Whether the nodes of the graph are stored in degree-bucket order

  heap_profiler::unique_ptr<std::uint8_t> _compressed_edges;
  EdgeID _compressed_edges_size;
  EdgeID _num_edges;

  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  NodeID _max_degree;
  NodeWeight _total_node_weight;
  EdgeWeight _total_edge_weight;

  // Statistics about graph compression
  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;
};

} // namespace kaminpar::shm
