/*******************************************************************************
 * Implements common synchronization operations for distributed graphs.
 *
 * @file:   graph_synchronization.h
 * @author: Daniel Seemaier
 * @date:   15.07.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"

namespace kaminpar::dist::graph {
/*!
 * Synchronizes the block assignment of ghost nodes: each node sends its current
 * assignment to all replicates (ghost nodes) residing on other PEs.
 *
 * @param p_graph Graph partition to synchronize.
 */
void synchronize_ghost_node_block_ids(DistributedPartitionedGraph &p_graph);

void synchronize_ghost_node_weights(DistributedGraph &graph);
} // namespace kaminpar::dist::graph
