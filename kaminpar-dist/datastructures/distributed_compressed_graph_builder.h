/*******************************************************************************
 * Sequential builder for distributed compressed graphs.
 *
 * @file:   distributed_compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   07.06.2024
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph-compression/compressed_edges_builder.h"

namespace kaminpar::dist {

/*!
 * A sequential builder that constructs compressed graphs.
 */
class DistributedCompressedGraphBuilder {
public:
  [[nodiscard]] static DistributedCompressedGraph compress(const DistributedCSRGraph &graph);

  /*!
   * Constructs a new DistributedCompressedGraphBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes to add are stored in degree-bucket order.
   */
  DistributedCompressedGraphBuilder(
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
   * Builds the compressed graph. The builder must then be reinitialized in order to compress
   * another graph.
   *
   * @return The components of the compressed graph that has been build.
   */
  std::tuple<StaticArray<EdgeID>, CompressedEdges<NodeID, EdgeID>, StaticArray<EdgeWeight>> build();

private:
  bool _sorted; // Whether the nodes of the graph are stored in degree-bucket order
  StaticArray<EdgeID> _nodes;

  EdgeID _num_edges;
  CompressedEdgesBuilder<NodeID, EdgeID, EdgeWeight> _compressed_edges_builder;
  StaticArray<EdgeWeight> _edge_weights;
};

} // namespace kaminpar::dist
