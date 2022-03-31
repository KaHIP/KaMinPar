/*******************************************************************************
 * @file:   factories.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Factory functions to instantiate coarsening and local improvement
 * algorithms.
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/i_coarsener.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/initial_partitioning/initial_refiner.h"
#include "kaminpar/refinement/i_balancer.h"
#include "kaminpar/refinement/i_refiner.h"

namespace kaminpar::factory {
std::unique_ptr<ICoarsener> create_coarsener(const Graph& graph, const CoarseningContext& c_ctx);

std::unique_ptr<ip::InitialRefiner> create_initial_refiner(
    const Graph& graph, const PartitionContext& p_ctx, const RefinementContext& r_ctx,
    ip::InitialRefiner::MemoryContext m_ctx = {});

std::unique_ptr<IRefiner> create_refiner(const Context& ctx);

std::unique_ptr<IBalancer>
create_balancer(const Graph& graph, const PartitionContext& p_ctx, const RefinementContext& r_ctx);
} // namespace kaminpar::factory
