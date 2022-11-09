/***********************************************************************************************************************
 * @file:   colored_lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds determined by a graph coloring.
 **********************************************************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class ColoredLPRefiner : public Refiner {
public:
    ColoredLPRefiner(const Context& ctx);

    ColoredLPRefiner(const ColoredLPRefiner&)            = delete;
    ColoredLPRefiner& operator=(const ColoredLPRefiner&) = delete;
    ColoredLPRefiner(ColoredLPRefiner&&) noexcept        = default;
    ColoredLPRefiner& operator=(ColoredLPRefiner&&)      = delete;

    void initialize(const DistributedGraph& graph) final;
    void refine(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) final;
};
} // namespace kaminpar::dist
