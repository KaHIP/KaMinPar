/*******************************************************************************
 * Interface for graph partitioning schemes.
 *
 * @file:   partitioner.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
class Partitioner {
public:
  virtual ~Partitioner() = default;
  virtual DistributedPartitionedGraph partition() = 0;
};
} // namespace kaminpar::dist
