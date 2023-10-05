/*******************************************************************************
 * Pseudo-refiner that does nothing.
 *
 * @file:   noop_refiner.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {
class NoopRefinerFactory : public GlobalRefinerFactory {
public:
  NoopRefinerFactory() = default;

  NoopRefinerFactory(const NoopRefinerFactory &) = delete;
  NoopRefinerFactory &operator=(const NoopRefinerFactory &) = delete;

  NoopRefinerFactory(NoopRefinerFactory &&) noexcept = default;
  NoopRefinerFactory &operator=(NoopRefinerFactory &&) = default;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;
};

class NoopRefiner : public GlobalRefiner {
public:
  NoopRefiner() = default;

  NoopRefiner(const NoopRefiner &) = delete;
  NoopRefiner &operator=(const NoopRefiner &) = delete;

  NoopRefiner(NoopRefiner &&) noexcept = default;
  NoopRefiner &operator=(NoopRefiner &&) = default;

  void initialize() final;
  bool refine() final;
};
} // namespace kaminpar::dist
