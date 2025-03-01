/*******************************************************************************
 * Instanties partitioning components specified by the Context struct.
 *
 * @file:   factories.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/coarsening/coarsener.h"
#include "kaminpar-dist/dkaminpar.h"
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

template <typename GainCache>
std::unique_ptr<GlobalRefinerFactory> create_refiner_with_decoupled_gain_cache(
    const Context &ctx, RefinementAlgorithm algorithm, GainCache &gain_cache
);

std::unique_ptr<Coarsener> create_coarsener(const Context &ctx);

std::unique_ptr<Clusterer> create_clusterer(const Context &ctx, ClusteringAlgorithm algorithm);

std::unique_ptr<Clusterer> create_clusterer(const Context &ctx);

} // namespace kaminpar::dist::factory
