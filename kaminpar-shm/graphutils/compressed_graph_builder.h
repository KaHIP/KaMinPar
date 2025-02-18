/*******************************************************************************
 * Sequential builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

namespace kaminpar::shm {

/*!
 * A sequential builder that constructs compressed graphs.
 */
class CompressedGraphBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;

  using CompressedNeighborhoodsBuilder =
      kaminpar::CompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight>;

public:
  /*!
   * Compresses a graph which is stored in compressed sparse row format.
   *
   * @param graph The graph to compress.
   * @return The compressed input graph.
   */
  static CompressedGraph compress(const CSRGraph &graph);

  /*!
   * Constructs a new CompressedGraphBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes that are added are stored in degree-bucket order.
   */
  CompressedGraphBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
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
  bool _sorted;
  CompressedNeighborhoodsBuilder _compressed_neighborhoods_builder;

  bool _store_node_weights;
  std::int64_t _total_node_weight;
  StaticArray<NodeWeight> _node_weights;
};

} // namespace kaminpar::shm
