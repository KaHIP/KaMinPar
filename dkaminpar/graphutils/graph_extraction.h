/*******************************************************************************
 * @file:   graph_extraction.h
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "kaminpar/datastructure/graph.h"

#include <vector>

namespace dkaminpar::graph {
struct ExtractedSubgraphs {
    std::vector<EdgeID>      shared_nodes;
    std::vector<NodeWeight>  shared_node_weights;
    std::vector<NodeID>      shared_edges;
    std::vector<EdgeWeight>  shared_edge_weights;
    std::vector<std::size_t> nodes_offset;
    std::vector<std::size_t> edges_offset;
    std::vector<NodeID>      mapping;
};

// Build a local block-induced subgraph for each block of the graph partition.
ExtractedSubgraphs
extract_local_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph, ExtractedSubgraphs memory = {});

std::vector<shm::Graph> distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph);
} // namespace dkaminpar::graph
