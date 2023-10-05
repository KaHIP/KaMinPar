/*******************************************************************************
 * Distributed label propagation refiner.
 *
 * @file:   lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {
class LPRefinerFactory : public GlobalRefinerFactory {
public:
  LPRefinerFactory(const Context &ctx);

  LPRefinerFactory(const LPRefinerFactory &) = delete;
  LPRefinerFactory &operator=(const LPRefinerFactory &) = delete;

  LPRefinerFactory(LPRefinerFactory &&) noexcept = default;
  LPRefinerFactory &operator=(LPRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class LPRefiner : public GlobalRefiner {
public:
  LPRefiner(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  LPRefiner(const LPRefiner &) = delete;
  LPRefiner &operator=(const LPRefiner &) = delete;

  LPRefiner(LPRefiner &&) noexcept = default;
  LPRefiner &operator=(LPRefiner &&) = delete;

  ~LPRefiner();

  void initialize() final;
  bool refine() final;

private:
  std::unique_ptr<class LPRefinerImpl> _impl;

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;
};
} // namespace kaminpar::dist
