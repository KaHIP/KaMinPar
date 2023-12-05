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

namespace kaminpar::dist {
std::unique_ptr<shm::Graph> allgather_graph(const DistributedGraph &graph);

std::pair<std::unique_ptr<shm::Graph>, std::unique_ptr<shm::PartitionedGraph>>
allgather_graph(const DistributedPartitionedGraph &p_graph);

shm::Graph replicate_graph_everywhere(const DistributedGraph &graph);

DistributedGraph replicate_graph(const DistributedGraph &graph, int num_replications);

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph &dist_graph, DistributedPartitionedGraph p_graph);

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph &dist_graph, shm::PartitionedGraph shm_p_graph);

DistributedPartitionedGraph distribute_partition(
    const DistributedGraph &graph,
    BlockID k,
    const StaticArray<shm::BlockID> &global_partition,
    PEID root
);
} // namespace kaminpar::dist
