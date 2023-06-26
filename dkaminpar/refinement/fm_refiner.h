/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 * @brief:  Distributed FM refiner.
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

#include "common/logger.h"
#include "common/parallel/atomic.h"

namespace kaminpar::dist {
class FMRefinerFactory : public GlobalRefinerFactory {
public:
  FMRefinerFactory(const Context &ctx);

  FMRefinerFactory(const FMRefinerFactory &) = delete;
  FMRefinerFactory &operator=(const FMRefinerFactory &) = delete;

  FMRefinerFactory(FMRefinerFactory &&) noexcept = default;
  FMRefinerFactory &operator=(FMRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class FMRefiner : public GlobalRefiner {
  SET_STATISTICS(true);
  SET_DEBUG(false);

public:
  FMRefiner(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;
  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  const FMRefinementContext &_fm_ctx;

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;
};
} // namespace kaminpar::dist
