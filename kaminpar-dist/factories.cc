/*******************************************************************************
 * Instanties partitioning components specified by the Context struct.
 *
 * @file:   factories.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#include "kaminpar-dist/factories.h"

#include <memory>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

// Partitioning schemes
#include "kaminpar-dist/partitioning/deep_multilevel.h"
#include "kaminpar-dist/partitioning/kway_multilevel.h"

// Initial Partitioning
#include "kaminpar-dist/initial_partitioning/kaminpar_initial_partitioner.h"
#include "kaminpar-dist/initial_partitioning/mtkahypar_initial_partitioner.h"
#include "kaminpar-dist/initial_partitioning/random_initial_partitioner.h"

// Refinement
#include "kaminpar-dist/refinement/adapters/mtkahypar_refiner.h"
#include "kaminpar-dist/refinement/balancer/cluster_balancer.h"
#include "kaminpar-dist/refinement/balancer/node_balancer.h"
#include "kaminpar-dist/refinement/jet/jet_refiner.h"
#include "kaminpar-dist/refinement/lp/clp_refiner.h"
#include "kaminpar-dist/refinement/lp/lp_refiner.h"
#include "kaminpar-dist/refinement/multi_refiner.h"
#include "kaminpar-dist/refinement/noop_refiner.h"

// Clustering
#include "kaminpar-dist/coarsening/clustering/hem/hem_clusterer.h"
#include "kaminpar-dist/coarsening/clustering/hem/hem_lp_clusterer.h"
#include "kaminpar-dist/coarsening/clustering/lp/global_lp_clusterer.h"
#include "kaminpar-dist/coarsening/clustering/lp/local_lp_clusterer.h"
#include "kaminpar-dist/coarsening/clustering/noop_clusterer.h"

namespace kaminpar::dist::factory {
std::unique_ptr<Partitioner>
create_partitioner(const Context &ctx, const DistributedGraph &graph, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return std::make_unique<DeepMultilevelPartitioner>(graph, ctx);

  case PartitioningMode::KWAY:
    return std::make_unique<KWayMultilevelPartitioner>(graph, ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<Partitioner> create_partitioner(const Context &ctx, const DistributedGraph &graph) {
  return create_partitioner(ctx, graph, ctx.mode);
}

std::unique_ptr<InitialPartitioner>
create_initial_partitioner(const Context &ctx, const InitialPartitioningAlgorithm algorithm) {
  switch (algorithm) {
  case InitialPartitioningAlgorithm::KAMINPAR:
    return std::make_unique<KaMinParInitialPartitioner>(ctx);

  case InitialPartitioningAlgorithm::MTKAHYPAR:
    return std::make_unique<MtKaHyParInitialPartitioner>(ctx);

  case InitialPartitioningAlgorithm::RANDOM:
    return std::make_unique<RandomInitialPartitioner>();
  }

  __builtin_unreachable();
}

std::unique_ptr<InitialPartitioner> create_initial_partitioner(const Context &ctx) {
  return create_initial_partitioner(ctx, ctx.initial_partitioning.algorithm);
}

std::unique_ptr<GlobalRefinerFactory>
create_refiner(const Context &ctx, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return std::make_unique<NoopRefinerFactory>();

  case RefinementAlgorithm::BATCHED_LP_REFINER:
    return std::make_unique<LPRefinerFactory>(ctx);

  case RefinementAlgorithm::COLORED_LP_REFINER:
    return std::make_unique<ColoredLPRefinerFactory>(ctx);

  case RefinementAlgorithm::JET_REFINER:
    return std::make_unique<JetRefinerFactory>(ctx);

  case RefinementAlgorithm::HYBRID_NODE_BALANCER:
    return std::make_unique<NodeBalancerFactory>(ctx);

  case RefinementAlgorithm::HYBRID_CLUSTER_BALANCER:
    return std::make_unique<ClusterBalancerFactory>(ctx);

  case RefinementAlgorithm::MTKAHYPAR_REFINER:
    return std::make_unique<MtKaHyParRefinerFactory>(ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<GlobalRefinerFactory> create_refiner(const Context &ctx) {
  if (ctx.refinement.algorithms.size() == 1) {
    return create_refiner(ctx, ctx.refinement.algorithms.front());
  }

  std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefinerFactory>> factories;
  for (const RefinementAlgorithm algorithm : ctx.refinement.algorithms) {
    if (factories.find(algorithm) == factories.end()) {
      factories[algorithm] = create_refiner(ctx, algorithm);
    }
  }

  return std::make_unique<MultiRefinerFactory>(std::move(factories), ctx.refinement.algorithms);
}

std::unique_ptr<GlobalClusterer>
create_global_clusterer(const Context &ctx, const GlobalClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case GlobalClusteringAlgorithm::NOOP:
    return std::make_unique<GlobalNoopClustering>(ctx);

  case GlobalClusteringAlgorithm::LP:
    return std::make_unique<GlobalLPClusterer>(ctx);

  case GlobalClusteringAlgorithm::HEM:
    return std::make_unique<HEMClusterer>(ctx);

  case GlobalClusteringAlgorithm::HEM_LP:
    return std::make_unique<HEMLPClusterer>(ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<GlobalClusterer> create_global_clusterer(const Context &ctx) {
  return create_global_clusterer(ctx, ctx.coarsening.global_clustering_algorithm);
}

std::unique_ptr<LocalClusterer>
create_local_clusterer(const Context &ctx, const LocalClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case LocalClusteringAlgorithm::NOOP:
    return std::make_unique<LocalNoopClustering>(ctx);

  case LocalClusteringAlgorithm::LP:
    return std::make_unique<LocalLPClusterer>(ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<LocalClusterer> create_local_clusterer(const Context &ctx) {
  return create_local_clusterer(ctx, ctx.coarsening.local_clustering_algorithm);
}
} // namespace kaminpar::dist::factory
