/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {
class LabelPropagationRefiner : public Refiner {
public:
  LabelPropagationRefiner(const Context &ctx);
  ~LabelPropagationRefiner();

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  class LabelPropagationRefinerImpl *_impl;
};
} // namespace kaminpar::shm
