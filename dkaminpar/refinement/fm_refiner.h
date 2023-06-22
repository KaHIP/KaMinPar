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
class FMRefiner : public Refiner {
  SET_STATISTICS(true);
  SET_DEBUG(false);

public:
  FMRefiner(const Context &ctx);

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;
  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize(const DistributedGraph &graph) final;
  void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  /*
   * Initialized by constructor
   */
  const FMRefinementContext &_fm_ctx;

  /*
   * Initialized by initialize()
   */
  const PartitionContext *_p_ctx;

  /*
   * Initialized by refine()
   */
  DistributedPartitionedGraph *_p_graph;
};
} // namespace kaminpar::dist
