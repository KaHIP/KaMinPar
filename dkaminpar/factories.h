/*******************************************************************************
 * @file:   factories.h
 *
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Instantiates the configured partitioning components.
 ******************************************************************************/
#pragma once

#include <memory>

#include "dkaminpar/coarsening/i_clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/initial_partitioning/i_initial_partitioner.h"
#include "dkaminpar/refinement/i_distributed_refiner.h"

namespace dkaminpar::factory {
std::unique_ptr<IInitialPartitioner>               create_initial_partitioning_algorithm(const Context& ctx);
std::unique_ptr<IDistributedRefiner>               create_refinement_algorithm(const Context& ctx);
std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>> create_global_clustering_algorithm(const Context& ctx);
std::unique_ptr<ClusteringAlgorithm<NodeID>>       create_local_clustering_algorithm(const Context& ctx);
} // namespace dkaminpar::factory
