/*******************************************************************************
 * @file:   i_distributed_refiner.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for refinement algorithms.
 ******************************************************************************/

#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_context.h"

namespace dkaminpar {
class IDistributedRefiner {
public:
  virtual ~IDistributedRefiner() = default;

  virtual void initialize(const DistributedGraph &graph, const PartitionContext &p_ctx) = 0;
  virtual void refine(DistributedPartitionedGraph &p_graph) = 0;
};
} // namespace dkaminpar