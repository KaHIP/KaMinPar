/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class LabelPropagationRefiner : public Refiner {
public:
  LabelPropagationRefiner(const Context &ctx);

  LabelPropagationRefiner(const LabelPropagationRefiner &) = delete;
  LabelPropagationRefiner &operator=(const LabelPropagationRefiner &) = delete;

  LabelPropagationRefiner(LabelPropagationRefiner &&) noexcept = default;
  LabelPropagationRefiner &operator=(LabelPropagationRefiner &&) noexcept = default;

  ~LabelPropagationRefiner() override;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

  void set_communities(std::span<const NodeID> communities) override;

private:
  std::unique_ptr<class LPRefinerImplWrapper> _impl_wrapper;
};

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
