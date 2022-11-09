/***********************************************************************************************************************
 * @file:   colored_lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds determined by a graph coloring.
 **********************************************************************************************************************/
#include "dkaminpar/refinement/colored_lp_refiner.h"

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
ColoredLPRefiner::ColoredLPRefiner(const Context& ctx) {
    ((void)ctx);
}

void ColoredLPRefiner::initialize(const DistributedGraph& graph) {
    ((void)graph);
}

void ColoredLPRefiner::refine(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    ((void)p_graph);
    ((void)p_ctx);
}
} // namespace kaminpar::dist
