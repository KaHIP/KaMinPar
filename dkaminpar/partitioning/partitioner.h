/*******************************************************************************
 * @file:   partitioner.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for graph partitioning schemes.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
class Partitioner {
public:
  virtual ~Partitioner() = default;

  virtual DistributedPartitionedGraph partition() = 0;
};
} // namespace kaminpar::dist
