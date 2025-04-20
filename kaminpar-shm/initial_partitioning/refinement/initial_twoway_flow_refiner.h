/*******************************************************************************
 * Sequential 2-way flow refinement used during initial bipartitioning.
 *
 * @file:   initial_twoway_flow_refiner.h
 * @author: Daniel Sallwasser
 * @date:   20.04.2025
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/initial_partitioning/refinement/initial_refiner.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"

namespace kaminpar::shm {

class InitialTwowayFlowRefiner : public InitialRefiner {
public:
  explicit InitialTwowayFlowRefiner(const TwowayFlowRefinementContext &f_ctx);

  void init(const CSRGraph &graph) final;

  bool refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  TwowayFlowRefiner _refiner;

  const CSRGraph *_graph;
};

} // namespace kaminpar::shm
