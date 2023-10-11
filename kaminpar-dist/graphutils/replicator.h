/*******************************************************************************
 * Allgather a distributed graph to each PE.
 *
 * @file:   allgather_graph.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <mpi.h>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::dist::graph {
shm::Graph replicate_everywhere(const DistributedGraph &graph);

DistributedGraph replicate(const DistributedGraph &graph, int num_replications);

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph &dist_graph, DistributedPartitionedGraph p_graph);

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph &dist_graph, shm::PartitionedGraph shm_p_graph);
} // namespace kaminpar::dist::graph