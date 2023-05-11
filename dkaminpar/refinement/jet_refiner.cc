/*******************************************************************************
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 * @brief:  Distributed JET refiner.
 ******************************************************************************/
#include "dkaminpar/refinement/jet_refiner.h"

#include "dkaminpar/context.h"

namespace kaminpar::dist {
JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

void JetRefiner::refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  ((void)p_graph);
  ((void)p_ctx);
}
} // namespace kaminpar::dist
