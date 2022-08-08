/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.08.2022
 * @brief:  Distributed FM refiner.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/refinement/i_distributed_refiner.h"

namespace kaminpar::dist {
class FMRefiner : public IDistributedRefiner {
public:
    FMRefiner(const Context& ctx);

    FMRefiner(const FMRefiner&)            = delete;
    FMRefiner(FMRefiner&&)                 = default;
    FMRefiner& operator=(const FMRefiner&) = delete;
    FMRefiner& operator=(FMRefiner&&)      = delete;

    void initialize(const DistributedGraph& graph, const PartitionContext& p_ctx);
    void refine(DistributedPartitionedGraph& p_graph);

private:
    tbb::concurrent_vector<NodeID> find_seed_nodes();

    // initialized by ctor
    const PartitionContext&    _p_ctx;
    const FMRefinementContext& _fm_ctx;

    // initalized by refine()
    DistributedPartitionedGraph* _p_graph;

    // initialized here
    std::size_t _iteration{0};
};
} // namespace kaminpar::dist
