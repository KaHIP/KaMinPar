/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#pragma once

#include <memory>
#include <string>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/scheduler/parallel_active_block_scheduler.h"
#include "kaminpar-shm/refinement/flow/scheduler/sequential_active_block_scheduler.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class TwowayFlowRefiner : public Refiner {
public:
  TwowayFlowRefiner(const ParallelContext &par_ctx, const TwowayFlowRefinementContext &f_ctx);
  ~TwowayFlowRefiner() override = default;

  TwowayFlowRefiner(TwowayFlowRefiner &&) noexcept = default;
  TwowayFlowRefiner &operator=(TwowayFlowRefiner &&) noexcept = delete;

  TwowayFlowRefiner(const TwowayFlowRefiner &) = delete;
  TwowayFlowRefiner &operator=(const TwowayFlowRefiner &) = delete;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx);

private:
  const ParallelContext &_par_ctx;
  const TwowayFlowRefinementContext &_f_ctx;

  std::unique_ptr<SequentialActiveBlockScheduler> _sequential_scheduler;
  std::unique_ptr<ParallelActiveBlockScheduler> _parallel_scheduler;
};

} // namespace kaminpar::shm
