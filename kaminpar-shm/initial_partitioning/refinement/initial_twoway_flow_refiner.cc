#include "kaminpar-shm/initial_partitioning/refinement/initial_twoway_flow_refiner.h"

namespace kaminpar::shm {

InitialTwowayFlowRefiner::InitialTwowayFlowRefiner(const TwowayFlowRefinementContext &f_ctx)
    : _refiner(f_ctx) {}

void InitialTwowayFlowRefiner::init(const CSRGraph &graph) {
  _graph = &graph;
}

bool InitialTwowayFlowRefiner::refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) {
  return _refiner.refine(p_graph, *_graph, p_ctx);
}

} // namespace kaminpar::shm
