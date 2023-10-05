/*******************************************************************************
 * Instanties partitioning components specified by the Context struct.
 *
 * @file:   factories.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/initial_partitioning/initial_partitioner.h"
#include "kaminpar-dist/partitioning/partitioner.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist::factory {
std::unique_ptr<Partitioner>
create_partitioner(const Context &ctx, const DistributedGraph &graph, PartitioningMode mode);

std::unique_ptr<Partitioner> create_partitioner(const Context &ctx, const DistributedGraph &graph);

std::unique_ptr<InitialPartitioner>
create_initial_partitioner(const Context &ctx, InitialPartitioningAlgorithm algorithm);

std::unique_ptr<InitialPartitioner> create_initial_partitioner(const Context &ctx);

std::unique_ptr<GlobalRefinerFactory>
create_refiner(const Context &ctx, RefinementAlgorithm algorithm);

std::unique_ptr<GlobalRefinerFactory> create_refiner(const Context &ctx);

std::unique_ptr<GlobalClusterer>
create_global_clusterer(const Context &ctx, GlobalClusteringAlgorithm algorithm);

std::unique_ptr<GlobalClusterer> create_global_clusterer(const Context &ctx);

std::unique_ptr<LocalClusterer>
create_local_clusterer(const Context &ctx, LocalClusteringAlgorithm algorithm);

std::unique_ptr<LocalClusterer> create_local_clusterer(const Context &ctx);
} // namespace kaminpar::dist::factory
