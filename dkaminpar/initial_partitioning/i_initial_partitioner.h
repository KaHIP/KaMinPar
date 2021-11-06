/*******************************************************************************
 * @file:   i_initial_partitioner.h
 *
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/distributed_definitions.h"
#include "kaminpar/datastructure/graph.h"

namespace dkaminpar {
class IInitialPartitioner {
public:
  virtual ~IInitialPartitioner() = default;
  virtual shm::PartitionedGraph initial_partition(const shm::Graph &graph) = 0;
};
} // namespace dkaminpar