#include "kaminpar-shm/initial_partitioning/refinement/initial_twoway_flow_refiner.h"

namespace kaminpar::shm {

InitialTwowayFlowRefiner::InitialTwowayFlowRefiner(const TwowayFlowRefinementContext &f_ctx)
    : _par_ctx(1),
      _refiner(_par_ctx, f_ctx) {
  KASSERT(f_ctx.run_sequentially);
  KASSERT(!f_ctx.scheduler.parallel);
}

void InitialTwowayFlowRefiner::init(const CSRGraph &graph) {
  _graph = &graph;
}

bool InitialTwowayFlowRefiner::refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) {
  return _refiner.refine(p_graph, *_graph, p_ctx);
}

} // namespace kaminpar::shm
