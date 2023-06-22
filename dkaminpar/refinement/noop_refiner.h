/*******************************************************************************
 * @file:   noop_refiner.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Refiner that does nothing.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class NoopRefiner : public Refiner {
public:
  NoopRefiner() = default;

  NoopRefiner(const NoopRefiner &) = delete;
  NoopRefiner &operator=(const NoopRefiner &) = delete;
  NoopRefiner(NoopRefiner &&) noexcept = default;
  NoopRefiner &operator=(NoopRefiner &&) = delete;

  void initialize(const DistributedGraph &graph) final;
  void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;
};
} // namespace kaminpar::dist
