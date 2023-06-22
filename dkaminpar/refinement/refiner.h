/*******************************************************************************
 * @file:   refiner.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for refinement algorithms.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
class Refiner {
public:
  virtual ~Refiner() = default;

  virtual void initialize(const DistributedGraph &graph) = 0;
  virtual void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;
};
} // namespace kaminpar::dist
