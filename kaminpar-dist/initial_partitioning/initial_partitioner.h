/*******************************************************************************
 * Interface for initial partitionign algorithms.
 *
 * @file:   initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   30.09.21
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::dist {

class InitialPartitioner {
public:
  virtual ~InitialPartitioner() = default;

  virtual shm::PartitionedGraph
  initial_partition(const shm::Graph &graph, const shm::PartitionContext &p_ctx) = 0;
};

} // namespace kaminpar::dist
