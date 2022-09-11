/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 * @brief:  Distributed FM refiner.
 ******************************************************************************/
#include "dkaminpar/refinement/fm_refiner.h"

#include "dkaminpar/context.h"
#include "dkaminpar/graphutils/bfs_extractor.h"
#include "dkaminpar/graphutils/independent_set.h"

namespace kaminpar::dist {
FMRefiner::FMRefiner(const Context& ctx) : _fm_ctx(ctx.refinement.fm) {}

void FMRefiner::initialize(const DistributedGraph&, const PartitionContext& p_ctx) {
    _p_ctx = &p_ctx;
}

void FMRefiner::refine(DistributedPartitionedGraph& p_graph) {
    _p_graph = &p_graph;
}
} // namespace kaminpar::dist
