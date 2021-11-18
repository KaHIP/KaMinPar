/*******************************************************************************
 * @file:   allgather_graph.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Allgather a distributed graph to each PE.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "kaminpar/datastructure/graph.h"

namespace dkaminpar::graph {
shm::Graph allgather(const DistributedGraph &graph);
DistributedPartitionedGraph reduce_scatter(const DistributedGraph &dist_graph, shm::PartitionedGraph shm_p_graph);
} // namespace dkaminpar::graph