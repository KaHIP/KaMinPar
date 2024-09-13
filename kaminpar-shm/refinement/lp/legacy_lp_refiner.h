/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class LegacyLabelPropagationRefiner : public Refiner {
public:
  LegacyLabelPropagationRefiner(const Context &ctx);

  ~LegacyLabelPropagationRefiner() override;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  class LegacyLabelPropagationRefinerImpl *_impl;
};

} // namespace kaminpar::shm
