/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class TwowayFlowRefiner : public Refiner {
public:
  TwowayFlowRefiner(const Context &ctx);
  ~TwowayFlowRefiner() override;

  TwowayFlowRefiner(const TwowayFlowRefiner &) = delete;
  TwowayFlowRefiner &operator=(const TwowayFlowRefiner &) = delete;

  TwowayFlowRefiner(TwowayFlowRefiner &&) noexcept = default;
  TwowayFlowRefiner &operator=(TwowayFlowRefiner &&) noexcept = default;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

  bool refine(PartitionedGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx);

private:
  const TwowayFlowRefinementContext &_f_ctx;
};

} // namespace kaminpar::shm
