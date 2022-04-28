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

#include <vector>

namespace dkaminpar::graph {
std::vector<DistributedGraph> distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph);
}
