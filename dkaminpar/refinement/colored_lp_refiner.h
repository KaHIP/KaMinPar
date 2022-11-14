/***********************************************************************************************************************
 * @file:   colored_lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds determined by a graph coloring.
 **********************************************************************************************************************/
#pragma once

#include "dkaminpar/algorithms/greedy_node_coloring.h"
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

private:
    void handle_node(NodeID u);

    void synchronize_state(ColorID c);

    const Context& _input_ctx;

    const PartitionContext*      _p_ctx;
    DistributedPartitionedGraph* _p_graph;

    NoinitVector<ColorID> _color_sizes;
    NoinitVector<NodeID>  _color_sorted_nodes;
};
} // namespace kaminpar::dist
