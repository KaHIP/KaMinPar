/*******************************************************************************
 * @file:   graph_extraction.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#pragma once

#include <vector>

#include "dkaminpar/datastructures/distributed_graph.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

namespace kaminpar::dist::graph {
struct ExtractedLocalSubgraphs {
  std::vector<EdgeID> shared_nodes;
  std::vector<NodeWeight> shared_node_weights;
  std::vector<NodeID> shared_edges;
  std::vector<EdgeWeight> shared_edge_weights;
  std::vector<std::size_t> nodes_offset;
  std::vector<std::size_t> edges_offset;
  std::vector<NodeID> mapping;
};

/*!
 * Extracts the block induced subgraph for each block.
 *
 * @param p_graph Partitioned graph from which the block induced subgraphs are
 * extracted.
 * @return For each block k, a data structure describing the graph induced by
 * all *local* nodes of p_graph in block k.
 */
ExtractedLocalSubgraphs
extract_local_block_induced_subgraphs(const DistributedPartitionedGraph &p_graph
);

struct ExtractedSubgraphs {
  /*!
   * Subgraphs assigned to this PE.
   */
  std::vector<shm::Graph> subgraphs;

  /*!
   * For each subgraph b, subgraph_offsets[b] is an array indicated which part
   * of the subgraphs is owned by which PE in the distributed graph. I.e., nodes
   * [subgraph_offset[b][i], subgraph_offset[b][i + 1]) of subgraph b are owned
   * by PE i.
   */
  std::vector<std::vector<NodeID>> subgraph_offsets;

  /*!
   * Mapping from nodes in the distributed graph to node IDs in the subgraph.
   */
  std::vector<NodeID> mapping;
};

/*!
 * Extracts all block induced subgraphs and distributes them to PEs.
 *
 * @param p_graph Partitioned graph from which the block induced subgraphs are
 * extracted.
 * @return Block induced subgraphs with meta data required to implement the
 * reverse operation.
 */
ExtractedSubgraphs extract_and_scatter_block_induced_subgraphs(
    const DistributedPartitionedGraph &p_graph
);

DistributedPartitionedGraph copy_subgraph_partitions(
    DistributedPartitionedGraph p_graph,
    const std::vector<shm::PartitionedGraph> &p_subgraphs,
    ExtractedSubgraphs &subgraphs
);

DistributedPartitionedGraph copy_duplicated_subgraph_partitions(
    DistributedPartitionedGraph p_graph,
    const std::vector<shm::PartitionedGraph> &p_subgraphs,
    ExtractedSubgraphs &extracted_subgraphs
);
} // namespace kaminpar::dist::graph
