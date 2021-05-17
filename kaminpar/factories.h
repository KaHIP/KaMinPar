#pragma once

#include "coarsening/i_coarsener.h"
#include "context.h"
#include "datastructure/graph.h"
#include "initial_partitioning/i_bipartitioner.h"
#include "initial_partitioning/initial_refiner.h"
#include "refinement/i_balancer.h"
#include "refinement/i_refiner.h"

namespace kaminpar::factory {
std::unique_ptr<Coarsener> create_coarsener(const Graph &graph, const CoarseningContext &c_ctx);

std::unique_ptr<ip::InitialRefiner> create_initial_refiner(const Graph &graph, const PartitionContext &p_ctx,
                                                           const RefinementContext &r_ctx,
                                                           ip::InitialRefiner::MemoryContext m_ctx = {});

std::unique_ptr<Refiner> create_refiner(const Graph &graph, const PartitionContext &p_ctx,
                                        const RefinementContext &r_ctx);

std::unique_ptr<Balancer> create_balancer(const Graph &graph, const PartitionContext &p_ctx,
                                          const RefinementContext &r_ctx);
} // namespace kaminpar::factory