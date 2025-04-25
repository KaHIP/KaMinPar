/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.cc
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/multiway_flow_refiner.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
namespace {
SET_DEBUG(true);
}

MultiwayFlowRefiner::MultiwayFlowRefiner(const Context &ctx)
    : _p_ctx(ctx.partition),
      _f_ctx(ctx.refinement.multiway_flow) {}

MultiwayFlowRefiner::~MultiwayFlowRefiner() = default;

std::string MultiwayFlowRefiner::name() const {
  return "Flow Refinement";
}

void MultiwayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool MultiwayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) { return refine(p_graph, csr_graph); },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the multiway flow refiner.";
        return false;
      }
  );
}

bool MultiwayFlowRefiner::refine(PartitionedGraph &p_graph, const CSRGraph &graph) {
  SCOPED_TIMER("Multiway Flow Refinement");

  _p_graph = &p_graph;
  _graph = &graph;

  return false;
}

} // namespace kaminpar::shm
