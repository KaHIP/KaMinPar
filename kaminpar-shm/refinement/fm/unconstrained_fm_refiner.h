/*******************************************************************************
 * Parallel unconstrained k-way FM refinement algorithm.
 *
 * @file:   unconstrained_fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   20.05.2026
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class UnconstrainedFMRefiner : public Refiner {
public:
  UnconstrainedFMRefiner(const Context &ctx);

  // Note: requires dtor definition in the *.cc file due to the std::unique_ptr<> member.
  ~UnconstrainedFMRefiner() override;

  UnconstrainedFMRefiner(const UnconstrainedFMRefiner &) = delete;
  UnconstrainedFMRefiner &operator=(const UnconstrainedFMRefiner &) = delete;

  UnconstrainedFMRefiner(UnconstrainedFMRefiner &&) noexcept = default;
  UnconstrainedFMRefiner &operator=(UnconstrainedFMRefiner &&) = delete;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;

  std::unique_ptr<Refiner> _core;
};

} // namespace kaminpar::shm
