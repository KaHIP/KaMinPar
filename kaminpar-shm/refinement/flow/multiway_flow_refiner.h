/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class MultiwayFlowRefiner : public Refiner {

public:
  MultiwayFlowRefiner(const Context &ctx);
  ~MultiwayFlowRefiner() override;

  MultiwayFlowRefiner(const MultiwayFlowRefiner &) = delete;
  MultiwayFlowRefiner &operator=(const MultiwayFlowRefiner &) = delete;

  MultiwayFlowRefiner(MultiwayFlowRefiner &&) noexcept = default;
  MultiwayFlowRefiner &operator=(MultiwayFlowRefiner &&) noexcept = default;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  bool refine(PartitionedGraph &p_graph, const CSRGraph &graph);

private:
  const PartitionContext &_p_ctx;
  const MultiwayFlowRefinementContext &_f_ctx;

  PartitionedGraph *_p_graph;
  const CSRGraph *_graph;
};

} // namespace kaminpar::shm
