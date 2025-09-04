#include "kaminpar-shm/refinement/flow/scheduler/active_block_scheduler_base.h"

#include <limits>

#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/hyper_flow_cutter.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

void ActiveBlockSchedulerStatistics::reset() {
  num_searches = 0;
  num_local_improvements = 0;
  num_global_improvements = 0;
  num_move_conflicts = 0;
  num_imbalance_conflicts = 0;
  num_failed_imbalance_resolutions = 0;
  num_successful_imbalance_resolutions = 0;
  min_imbalance = std::numeric_limits<double>::max();
  max_imbalance = std::numeric_limits<double>::min();
  total_imbalance = 0.0;
}

void ActiveBlockSchedulerStatistics::print() const {
  LOG_STATS << "Two-Way Flow Refiner:";
  LOG_STATS << "*  # num searches: " << num_searches;
  LOG_STATS << "*  # num local improvements: " << num_local_improvements;
  LOG_STATS << "*  # num global improvements: " << num_global_improvements;
  LOG_STATS << "*  # num move conflicts: " << num_move_conflicts;
  LOG_STATS << "*  # num imbalance conflicts: " << num_imbalance_conflicts;
  LOG_STATS << "*  # num failed imbalance resolutions: " << num_failed_imbalance_resolutions;
  LOG_STATS << "*  # num successful imbalance resolutions: "
            << num_successful_imbalance_resolutions;
  LOG_STATS << "*  # min / average / max imbalance: "
            << (num_imbalance_conflicts ? min_imbalance : 0) << " / "
            << (num_imbalance_conflicts ? total_imbalance / num_imbalance_conflicts : 0) << " / "
            << (num_imbalance_conflicts ? max_imbalance : 0);
}

FlowRefiner::FlowRefiner(
    const PartitionContext &p_ctx,
    const TwowayFlowRefinementContext &f_ctx,
    const bool run_sequentially,
    const QuotientGraph &q_graph,
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    const TimePoint &start_time
)
    : _p_graph(p_graph),
      _run_sequentially(run_sequentially),
      _border_region_constructor(p_ctx, f_ctx.construction, q_graph, p_graph, graph),
      _flow_network_constructor(f_ctx.construction, run_sequentially, p_graph, graph) {
#ifdef KAMINPAR_WHFC_FOUND
  if (f_ctx.flow_cutter.use_whfc) {
    _flow_cutter_algorithm =
        std::make_unique<HyperFlowCutter>(p_ctx, f_ctx.flow_cutter, run_sequentially);
  } else {
    _flow_cutter_algorithm =
        std::make_unique<FlowCutter>(p_ctx, f_ctx.flow_cutter, run_sequentially, p_graph);
  }
#else
  if (f_ctx.flow_cutter.use_whfc) {
    LOG_WARNING << "WHFC requested but not available; using built-in FlowCutter as fallback.";
  }

  _flow_cutter_algorithm =
      std::make_unique<FlowCutter>(p_ctx, f_ctx.flow_cutter, run_sequentially, p_graph);
#endif

  _flow_cutter_algorithm->set_time_limit(f_ctx.time_limit, start_time);
}

FlowRefiner::Result FlowRefiner::refine(const BlockID block1, const BlockID block2) {
  KASSERT(block1 != block2, "The flow refiner can only work on distinct block pairs");
  SCOPED_TIMER("Refine Block Pair");

  const BlockWeight block_weight1 = _p_graph.block_weight(block1);
  const BlockWeight block_weight2 = _p_graph.block_weight(block2);

  const BorderRegion &border_region =
      _border_region_constructor.construct(block1, block2, block_weight1, block_weight2);

  const FlowNetwork flow_network =
      _flow_network_constructor.construct_flow_network(border_region, block_weight1, block_weight2);

  return _flow_cutter_algorithm->compute_cut(border_region, flow_network);
}

} // namespace kaminpar::shm
