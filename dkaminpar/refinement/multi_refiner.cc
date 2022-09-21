/*******************************************************************************
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 * @brief:  Pseudo-refiner that runs multiple refiners in sequence.
 ******************************************************************************/
#include "dkaminpar/refinement/multi_refiner.h"

namespace kaminpar::dist {
MultiRefiner::MultiRefiner(std::vector<std::unique_ptr<Refiner>> refiners) : _refiners(std::move(refiners)) {}

void MultiRefiner::initialize(const DistributedGraph& graph, const PartitionContext& p_ctx) {
    for (const auto& refiner: _refiners) {
        refiner->initialize(graph, p_ctx);
    }
}

void MultiRefiner::refine(DistributedPartitionedGraph& p_graph) {
    for (const auto& refiner: _refiners) {
        refiner->refine(p_graph);
    }
}
} // namespace kaminpar::dist
