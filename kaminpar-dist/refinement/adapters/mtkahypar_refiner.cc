/*******************************************************************************
 * Adapter to use Mt-KaHyPar as a refinement algorithm.
 *
 * @file:   mtkahypar_refiner.cc
 * @author: Daniel Seemaier
 * @date:   17.10.2023
 ******************************************************************************/
#include "kaminpar-dist/refinement/adapters/mtkahypar_refiner.h"

namespace kaminpar::dist {
MtKaHyParRefinerFactory::MtKaHyParRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner> MtKaHyParRefinerFactory::create(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  return std::make_unique<MtKaHyParRefiner>(_ctx, p_graph, p_ctx);
}

MtKaHyParRefiner::MtKaHyParRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx) {}

void MtKaHyParRefiner::initialize() {}

bool MtKaHyParRefiner::refine() {
  return false;
}
} // namespace kaminpar::dist
