/*******************************************************************************
 * @file:   graph_extraction.h
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#pragma once

#include <vector>

#include "dkaminpar/datastructure/distributed_graph.h"

#include "kaminpar/datastructure/graph.h"

namespace kaminpar::dist::graph {
struct ExtractedLocalSubgraphs {
    std::vector<EdgeID>      shared_nodes;
    std::vector<NodeWeight>  shared_node_weights;
    std::vector<NodeID>      shared_edges;
    std::vector<EdgeWeight>  shared_edge_weights;
    std::vector<std::size_t> nodes_offset;
    std::vector<std::size_t> edges_offset;
    std::vector<NodeID>      mapping;
};

ExtractedLocalSubgraphs extract_local_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph);

struct ExtractedSubgraphs {
    /*!
     * Completely local subgraphs assigned to this PE.
     */
    std::vector<shm::Graph> subgraphs;

    /*!
     * For each subgraph b, subgraph_offsets[b] is an array indicated which part of the subgraphs is owned by which PE
     * in the distributed graph. I.e., nodes [subgraph_offset[b][i], subgraph_offset[b][i + 1]) of subgraph b are owned
     * by PE i.
     */
    std::vector<std::vector<NodeID>> subgraph_offsets;

    /*!
     * Mapping from nodes in the distributed graph to node IDs in the subgraph.
     */
    std::vector<NodeID> mapping;
};

/*!
 * This operation builds a subgraph for each block of the partitioned graphs. The blocks are assigned to PE (each PE
 * gets the same number of blocks) and gathered, i.e., each PE will have an array of fully local graphs.
 *
 * @param p_graph The distributed, partitioned graph whose block-induced subgraphs are extracted and assigned to PEs.
 *
 * @return Extracted, local subgraphs along with some meta data required to implement the reverse operation, i.e.,
 * projecting partitions of the extracted subgraphs back to the distributed graph.
 */
ExtractedSubgraphs distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph);

DistributedPartitionedGraph copy_subgraph_partitions(
    DistributedPartitionedGraph p_graph, const std::vector<shm::PartitionedGraph>& p_subgraphs,
    const ExtractedSubgraphs& subgraphs);
} // namespace kaminpar::dist::graph
