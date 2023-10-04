/*******************************************************************************
 * Utility functions to extract and distribute block-induced subgraphs
 * of a partitioned distributed graph.
 *
 * These functions build an in-memory subgraph graph for each block of a
 * partitioned distributed graph. The blocks are assigned to PEs. If there are
 * fewer PEs than blocks, each PE gets multiple subgraphs, otherwise each
 * subgraph is duplicated and copied to multiple PEs.
 * The functions aim to distribute the subgraphs as evenly as possible.
 *
 * @file:   graph_extraction.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

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
extract_local_block_induced_subgraphs(const DistributedPartitionedGraph &p_graph);

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
ExtractedSubgraphs
extract_and_scatter_block_induced_subgraphs(const DistributedPartitionedGraph &p_graph);

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

class BlockExtractionOffsets {
public:
  BlockExtractionOffsets(PEID size, BlockID k);

  BlockID first_block_on_pe(PEID pe) const;
  BlockID first_invalid_block_on_pe(PEID pe) const;
  BlockID num_blocks_on_pe(PEID pe) const;

  PEID first_pe_with_block(BlockID block) const;
  PEID first_invalid_pe_with_block(BlockID block) const;
  PEID num_pes_with_block(BlockID block) const;

private:
  PEID _size;
  BlockID _k;
  BlockID _min_blocks_per_pe;
  BlockID _rem_blocks;
  PEID _min_pes_per_block;
  PEID _rem_pes;
};
} // namespace kaminpar::dist::graph
