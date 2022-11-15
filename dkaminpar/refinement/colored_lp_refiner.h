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

#include "common/parallel/vector_ets.h"

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
    using BlockGainsContainer = typename parallel::vector_ets<EdgeWeight>::Container;

    NodeID find_moves(ColorID c);
    NodeID perform_moves(ColorID c);
    NodeID attempt_moves(ColorID c, const BlockGainsContainer& block_gains);
    void   synchronize_state(ColorID c);

    void handle_node(NodeID u);

    const Context& _input_ctx;

    const PartitionContext*      _p_ctx;
    DistributedPartitionedGraph* _p_graph;

    NoinitVector<ColorID> _color_sizes;
    NoinitVector<NodeID>  _color_sorted_nodes;

    NoinitVector<BlockWeight> _block_weight_deltas;
    NoinitVector<EdgeWeight>  _gains;
    NoinitVector<BlockID>     _next_partition;
};
} // namespace kaminpar::dist
