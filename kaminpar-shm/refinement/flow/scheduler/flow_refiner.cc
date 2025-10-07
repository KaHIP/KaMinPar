#include "kaminpar-shm/refinement/flow/scheduler/flow_refiner.h"

#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/hyper_flow_cutter.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

FlowRefiner::FlowRefiner(
    const PartitionContext &p_ctx,
    const TwowayFlowRefinementContext &f_ctx,
    const QuotientGraph &q_graph,
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    const TimePoint &start_time
)
    : _p_graph(p_graph),
      _border_region_constructor(p_ctx, f_ctx.construction, q_graph, p_graph, graph),
      _flow_network_constructor(f_ctx.construction, p_graph, graph) {
#ifdef KAMINPAR_WHFC_FOUND
  if (f_ctx.flow_cutter.use_whfc) {
    _flow_cutter_algorithm = std::make_unique<HyperFlowCutter>(p_ctx, f_ctx.flow_cutter);
  } else {
    _flow_cutter_algorithm = std::make_unique<FlowCutter>(p_ctx, f_ctx.flow_cutter);
  }
#else
  if (f_ctx.flow_cutter.use_whfc) {
    LOG_WARNING << "WHFC requested but not available; using built-in FlowCutter as fallback.";
  }

  _flow_cutter_algorithm = std::make_unique<FlowCutter>(p_ctx, f_ctx.flow_cutter);
#endif

  _flow_cutter_algorithm->set_time_limit(f_ctx.time_limit, start_time);
}

FlowRefiner::Result
FlowRefiner::refine(const BlockID block1, const BlockID block2, const bool run_sequentially) {
  KASSERT(block1 != block2, "The flow refiner can only work on distinct block pairs");
  SCOPED_TIMER("Refine Block Pair");

  const BlockWeight block_weight1 = _p_graph.block_weight(block1);
  const BlockWeight block_weight2 = _p_graph.block_weight(block2);

  const BorderRegion &border_region =
      _border_region_constructor.construct(block1, block2, block_weight1, block_weight2);

  const FlowNetwork flow_network = _flow_network_constructor.construct_flow_network(
      border_region, block_weight1, block_weight2, run_sequentially
  );

  return _flow_cutter_algorithm->compute_cut(border_region, flow_network, run_sequentially);
}

void FlowRefiner::free() {
  _border_region_constructor.free();
  _flow_cutter_algorithm->free();
}

} // namespace kaminpar::shm
