/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

TwowayFlowRefiner::TwowayFlowRefiner(
    const ParallelContext &par_ctx, const TwowayFlowRefinementContext &f_ctx
)
    : _par_ctx(par_ctx),
      _f_ctx(f_ctx),
      _sequential_scheduler(std::make_unique<SequentialActiveBlockScheduler>(par_ctx, f_ctx)),
      _parallel_scheduler(std::make_unique<ParallelActiveBlockScheduler>(f_ctx)) {}

std::string TwowayFlowRefiner::name() const {
  return "Two-Way Flow Refinement";
}

void TwowayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool TwowayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) {
        // The flow refiner works with PartitionedCSRGraph instead of PartitionedGraph.
        // Intead of copying the partition, use a view on it to access the partition.
        PartitionedCSRGraph p_csr_graph = p_graph.csr_view();
        return refine(p_csr_graph, csr_graph, p_ctx);
      },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the two-way flow refiner.";
        return false;
      }
  );
}

bool TwowayFlowRefiner::refine(
    PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("Two-Way Flow Refinement");
  SCOPED_HEAP_PROFILER("Two-Way Flow Refinement");

  if (_f_ctx.scheduler.parallel && _par_ctx.num_threads > 1 && p_ctx.k > 2) {
    return _parallel_scheduler->refine(p_graph, graph, p_ctx);
  } else {
    return _sequential_scheduler->refine(p_graph, graph, p_ctx);
  }
}

} // namespace kaminpar::shm
