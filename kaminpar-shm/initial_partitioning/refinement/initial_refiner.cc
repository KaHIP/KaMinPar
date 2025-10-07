/*******************************************************************************
 * Interface for initial refinement algorithms.
 *
 * @file:   initial_refiner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/refinement/initial_refiner.h"

#include <utility>

#include "kaminpar-shm/initial_partitioning/refinement/initial_fm_refiner.h"
#include "kaminpar-shm/initial_partitioning/refinement/initial_noop_refiner.h"
#include "kaminpar-shm/initial_partitioning/refinement/initial_twoway_flow_refiner.h"

namespace kaminpar::shm {

InitialMultiRefiner::InitialMultiRefiner(
    std::unordered_map<InitialRefinementAlgorithm, std::unique_ptr<InitialRefiner>> refiners,
    std::vector<InitialRefinementAlgorithm> order
)
    : _refiners(std::move(refiners)),
      _order(std::move(order)) {}

void InitialMultiRefiner::init(const CSRGraph &graph) {
  _graph = &graph;
}

bool InitialMultiRefiner::refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) {
  bool found_improvement = false;

  for (InitialRefinementAlgorithm algorithm : _order) {
    InitialRefiner &refiner = *_refiners[algorithm];
    refiner.init(*_graph);

    const bool current_refiner_found_improvement = refiner.refine(p_graph, p_ctx);
    found_improvement |= current_refiner_found_improvement;
  }

  return found_improvement;
}

namespace {

std::unique_ptr<InitialRefiner> create_initial_refiner(
    const InitialRefinementContext &r_ctx, const InitialRefinementAlgorithm algorithm
) {
  switch (algorithm) {
  case InitialRefinementAlgorithm::NOOP:
    return std::make_unique<InitialNoopRefiner>();

  case InitialRefinementAlgorithm::TWOWAY_SIMPLE_FM:
    return std::make_unique<InitialSimple2WayFM>(r_ctx.fm);

  case InitialRefinementAlgorithm::TWOWAY_ADAPTIVE_FM:
    return std::make_unique<InitialAdaptive2WayFM>(r_ctx.fm);

  case InitialRefinementAlgorithm::TWOWAY_FLOW:
    return std::make_unique<InitialTwowayFlowRefiner>(r_ctx.twoway_flow);
  }

  __builtin_unreachable();
}

} // namespace

std::unique_ptr<InitialRefiner> create_initial_refiner(const InitialRefinementContext &r_ctx) {
  if (r_ctx.algorithms.empty()) {
    return std::make_unique<InitialNoopRefiner>();
  }

  if (r_ctx.algorithms.size() == 1) {
    return create_initial_refiner(r_ctx, r_ctx.algorithms.front());
  }

  std::unordered_map<InitialRefinementAlgorithm, std::unique_ptr<InitialRefiner>> refiners;
  for (const InitialRefinementAlgorithm algorithm : r_ctx.algorithms) {
    if (refiners.contains(algorithm)) {
      continue;
    }

    refiners[algorithm] = create_initial_refiner(r_ctx, algorithm);
  }

  return std::make_unique<InitialMultiRefiner>(std::move(refiners), r_ctx.algorithms);
}

} // namespace kaminpar::shm
