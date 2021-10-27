/*******************************************************************************
 * @file:   distributed_probabilistic_label_propagation_refiner.h
 *
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/refinement/i_distributed_refiner.h"

namespace dkaminpar {
class DistributedProbabilisticLabelPropagationRefiner : public IDistributedRefiner {
public:
  DistributedProbabilisticLabelPropagationRefiner(const Context &ctx);

  ~DistributedProbabilisticLabelPropagationRefiner();

  void initialize(const DistributedGraph &graph, const PartitionContext &p_ctx) override;

  void refine(DistributedPartitionedGraph &p_graph) override;

private:
  std::unique_ptr<class DistributedProbabilisticLabelPropagationRefinerImpl> _impl;
};
} // namespace dkaminpar