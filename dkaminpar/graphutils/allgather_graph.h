/*******************************************************************************
 * @file:   allgather_graph.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Allgather a distributed graph to each PE.
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "kaminpar/datastructures/graph.h"

namespace kaminpar::dist::graph {
shm::Graph                  allgather(const DistributedGraph& graph);
DistributedGraph            replicate(const DistributedGraph& graph, int num_replications);
DistributedPartitionedGraph reduce_scatter(const DistributedGraph& dist_graph, shm::PartitionedGraph shm_p_graph);
} // namespace kaminpar::dist::graph
