/*******************************************************************************
 * @file:   i_initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/definitions.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

namespace kaminpar::dist {
class InitialPartitioner {
public:
  virtual ~InitialPartitioner() = default;
  virtual shm::PartitionedGraph
  initial_partition(const shm::Graph &graph, const PartitionContext &p_ctx) = 0;
};
} // namespace kaminpar::dist
