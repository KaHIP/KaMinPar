/*******************************************************************************
 * Parallel k-way FM refinement algorithm.
 *
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/fm/fm_refiner.h"

#include "kaminpar-shm/refinement/fm/fm_refiner_core.h"

#include "kaminpar-common/console_io.h"

namespace kaminpar::shm {

FMRefiner::FMRefiner(const Context &input_ctx) : _ctx(input_ctx) {}
FMRefiner::~FMRefiner() = default;

std::string FMRefiner::name() const {
  return "FM";
}

void FMRefiner::initialize(const PartitionedGraph &p_graph) {
  _core = fm::create_fm_core(_ctx, p_graph);
  _core->initialize(p_graph);
}

bool FMRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  if (p_ctx.has_min_block_weights()) {
    LOG_WARNING << "FM refinement does not support min block weights. They will be ignored.";
  }

  return _core->refine(p_graph, p_ctx);
}

} // namespace kaminpar::shm
