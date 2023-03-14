/*******************************************************************************
 * @file:   label_propagation_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  K-way label propagation refinement algorithm.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm {
class LabelPropagationRefiner : public Refiner {
public:
  LabelPropagationRefiner(const Context &ctx);

  ~LabelPropagationRefiner();

  void initialize(const Graph &graph) override;

  bool refine(PartitionedGraph &p_graph,
              const PartitionContext &p_ctx) override;

  [[nodiscard]] EdgeWeight expected_total_gain() const override;

private:
  class LabelPropagationRefinerImpl *_impl;
};
} // namespace kaminpar::shm
