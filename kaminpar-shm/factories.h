/*******************************************************************************
 * Factory functions to instantiate partitioning composed based on their
 * respective enum constant.
 *
 * @file:   factories.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/partitioning/partitioner.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm::factory {

std::unique_ptr<Partitioner> create_partitioner(const Graph &graph, const ExecutionContext &ctx);

std::unique_ptr<Clusterer> create_clusterer(const ExecutionContext &ctx);

std::unique_ptr<Coarsener>
create_coarsener(const ExecutionContext &ctx, const PartitionContext &p_ctx);

std::unique_ptr<Coarsener> create_coarsener(const ExecutionContext &ctx);

std::unique_ptr<Refiner> create_refiner(const ExecutionContext &ctx);

} // namespace kaminpar::shm::factory
