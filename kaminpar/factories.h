/*******************************************************************************
 * Factory functions to instantiate partitioning composed based on their
 * respective enum constant.
 *
 * @file:   factories.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/coarsener.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/partitioning/partitioner.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm::factory {
std::unique_ptr<Partitioner> create_partitioner(const Graph &graph, const Context &ctx);
std::unique_ptr<Coarsener> create_coarsener(const Graph &graph, const CoarseningContext &c_ctx);
std::unique_ptr<Refiner> create_refiner(const Context &ctx);
} // namespace kaminpar::shm::factory
