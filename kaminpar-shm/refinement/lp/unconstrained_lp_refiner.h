/*******************************************************************************
 * Parallel unconstrained k-way label propagation refiner.
 *
 * @file:   unconstrained_lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   20.05.2026
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class UnconstrainedLabelPropagationRefiner : public Refiner {
public:
  UnconstrainedLabelPropagationRefiner(const Context &ctx);

  UnconstrainedLabelPropagationRefiner(const UnconstrainedLabelPropagationRefiner &) = delete;
  UnconstrainedLabelPropagationRefiner &
  operator=(const UnconstrainedLabelPropagationRefiner &) = delete;

  UnconstrainedLabelPropagationRefiner(UnconstrainedLabelPropagationRefiner &&) noexcept = default;
  UnconstrainedLabelPropagationRefiner &
  operator=(UnconstrainedLabelPropagationRefiner &&) noexcept = default;

  ~UnconstrainedLabelPropagationRefiner() override;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

  void set_communities(std::span<const NodeID> communities) override;

private:
  std::unique_ptr<class UnconstrainedLPRefinerImplWrapper> _impl_wrapper;
};

} // namespace kaminpar::shm
