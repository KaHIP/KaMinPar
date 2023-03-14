/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm {
class FMRefiner : public Refiner {
public:
  FMRefiner(const Context &ctx);

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) noexcept = default;

  void initialize(const Graph &graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  [[nodiscard]] EdgeWeight expected_total_gain() const final;
};
} // namespace kaminpar::shm
