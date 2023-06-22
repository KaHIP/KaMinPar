/*******************************************************************************
 * @file:   factories.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Instantiates the configured partitioning components.
 ******************************************************************************/
#pragma once

#include <memory>

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/initial_partitioning/initial_partitioner.h"
#include "dkaminpar/partitioning/partitioner.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist::factory {
std::unique_ptr<Partitioner> create_partitioner(const Context &ctx, const DistributedGraph &graph);
std::unique_ptr<InitialPartitioner> create_initial_partitioning_algorithm(const Context &ctx);
std::unique_ptr<Refiner> create_refinement_algorithm(const Context &ctx);
std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>>
create_global_clustering_algorithm(const Context &ctx);
std::unique_ptr<ClusteringAlgorithm<NodeID>> create_local_clustering_algorithm(const Context &ctx);
} // namespace kaminpar::dist::factory
