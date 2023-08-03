/*******************************************************************************
 * Instanties partitioning components specified by the Context struct.
 *
 * @file:   factories.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#include "dkaminpar/factories.h"

#include <memory>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/dkaminpar.h"

// Partitioning schemes
#include "dkaminpar/partitioning/deep_multilevel.h"
#include "dkaminpar/partitioning/kway_multilevel.h"

// Initial Partitioning
#include "dkaminpar/initial_partitioning/kaminpar_initial_partitioner.h"
#include "dkaminpar/initial_partitioning/mtkahypar_initial_partitioner.h"
#include "dkaminpar/initial_partitioning/random_initial_partitioner.h"

// Refinement
#include "dkaminpar/refinement/balancer/cluster_balancer.h"
#include "dkaminpar/refinement/balancer/greedy_balancer.h"
#include "dkaminpar/refinement/fm/fm_refiner.h"
#include "dkaminpar/refinement/fm/local_fm_refiner.h"
#include "dkaminpar/refinement/jet/jet_balancer.h"
#include "dkaminpar/refinement/jet/jet_refiner.h"
#include "dkaminpar/refinement/lp/clp_refiner.h"
#include "dkaminpar/refinement/lp/lp_refiner.h"
#include "dkaminpar/refinement/multi_refiner.h"
#include "dkaminpar/refinement/noop_refiner.h"

// Clustering
#include "dkaminpar/coarsening/clustering/hem/hem_clusterer.h"
#include "dkaminpar/coarsening/clustering/hem/hem_lp_clusterer.h"
#include "dkaminpar/coarsening/clustering/lp/global_lp_clusterer.h"
#include "dkaminpar/coarsening/clustering/lp/local_lp_clusterer.h"
#include "dkaminpar/coarsening/clustering/noop_clusterer.h"

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

  case RefinementAlgorithm::BATCHED_LP:
    return std::make_unique<LPRefinerFactory>(ctx);

  case RefinementAlgorithm::COLORED_LP:
    return std::make_unique<ColoredLPRefinerFactory>(ctx);

  case RefinementAlgorithm::LOCAL_FM:
    return std::make_unique<LocalFMRefinerFactory>(ctx);

  case RefinementAlgorithm::GLOBAL_FM:
    return std::make_unique<FMRefinerFactory>(ctx);

  case RefinementAlgorithm::JET_REFINER:
    return std::make_unique<JetRefinerFactory>(ctx);

  case RefinementAlgorithm::JET_BALANCER:
    return std::make_unique<JetBalancerFactory>(ctx);

  case RefinementAlgorithm::GREEDY_NODE_BALANCER:
    return std::make_unique<GreedyBalancerFactory>(ctx);

  case RefinementAlgorithm::GREEDY_CLUSTER_BALANCER:
    return std::make_unique<ClusterBalancerFactory>(ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<GlobalRefinerFactory> create_refiner(const Context &ctx) {
  if (ctx.refinement.algorithms.size() == 1) {
    return create_refiner(ctx, ctx.refinement.algorithms.front());
  }

  std::vector<std::unique_ptr<GlobalRefinerFactory>> factories;
  for (const RefinementAlgorithm algorithm : ctx.refinement.algorithms) {
    factories.push_back(create_refiner(ctx, algorithm));
  }
  return std::make_unique<MultiRefinerFactory>(std::move(factories));
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
