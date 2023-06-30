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
#include "dkaminpar/refinement/balancer/greedy_balancer.h"
#include "dkaminpar/refinement/balancer/move_set_balancer.h"
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
std::unique_ptr<Partitioner> create_partitioner(const Context &ctx, const DistributedGraph &graph) {
  switch (ctx.mode) {
  case PartitioningMode::DEEP:
    return std::make_unique<DeepMultilevelPartitioner>(graph, ctx);

  case PartitioningMode::KWAY:
    return std::make_unique<KWayPartitioner>(graph, ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<InitialPartitioner> create_initial_partitioner(const Context &ctx) {
  switch (ctx.initial_partitioning.algorithm) {
  case InitialPartitioningAlgorithm::KAMINPAR:
    return std::make_unique<KaMinParInitialPartitioner>(ctx);

  case InitialPartitioningAlgorithm::MTKAHYPAR:
    return std::make_unique<MtKaHyParInitialPartitioner>(ctx);

  case InitialPartitioningAlgorithm::RANDOM:
    return std::make_unique<RandomInitialPartitioner>();
  }

  __builtin_unreachable();
}

namespace {
std::unique_ptr<GlobalRefinerFactory>
create_refinement_algorithm(const Context &ctx, const KWayRefinementAlgorithm algorithm) {
  switch (algorithm) {
  case KWayRefinementAlgorithm::NOOP:
    return std::make_unique<NoopRefinerFactory>();

  case KWayRefinementAlgorithm::LP:
    return std::make_unique<LPRefinerFactory>(ctx);

  case KWayRefinementAlgorithm::LOCAL_FM:
    return std::make_unique<LocalFMRefinerFactory>(ctx);

  case KWayRefinementAlgorithm::FM:
    return std::make_unique<FMRefinerFactory>(ctx);

  case KWayRefinementAlgorithm::COLORED_LP:
    return std::make_unique<ColoredLPRefinerFactory>(ctx);

  case KWayRefinementAlgorithm::GREEDY_BALANCER:
    return std::make_unique<GreedyBalancerFactory>(ctx);

  case KWayRefinementAlgorithm::JET:
    return std::make_unique<JetRefinerFactory>(ctx);

  case KWayRefinementAlgorithm::MOVE_SET_BALANCER:
    return std::make_unique<MoveSetBalancerFactory>(ctx);

  case KWayRefinementAlgorithm::JET_BALANCER:
    return std::make_unique<JetBalancerFactory>(ctx);
  }

  __builtin_unreachable();
}
} // namespace

std::unique_ptr<GlobalRefinerFactory> create_refiner(const Context &ctx) {
  if (ctx.refinement.algorithms.size() == 1) {
    return create_refinement_algorithm(ctx, ctx.refinement.algorithms.front());
  }

  std::vector<std::unique_ptr<GlobalRefinerFactory>> factories;
  for (const KWayRefinementAlgorithm algorithm : ctx.refinement.algorithms) {
    factories.push_back(create_refinement_algorithm(ctx, algorithm));
  }
  return std::make_unique<MultiRefinerFactory>(std::move(factories));
}

std::unique_ptr<GlobalClusterer> create_global_clusterer(const Context &ctx) {
  switch (ctx.coarsening.global_clustering_algorithm) {
  case GlobalClusteringAlgorithm::NOOP:
    return std::make_unique<GlobalNoopClustering>(ctx);

  case GlobalClusteringAlgorithm::LP:
    return std::make_unique<GlobalLPClustering>(ctx);

  case GlobalClusteringAlgorithm::HEM:
    return std::make_unique<HEMClustering>(ctx);

  case GlobalClusteringAlgorithm::HEM_LP:
    return std::make_unique<HEMLPClustering>(ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<LocalClusterer> create_local_clusterer(const Context &ctx) {
  switch (ctx.coarsening.local_clustering_algorithm) {
  case LocalClusteringAlgorithm::NOOP:
    return std::make_unique<LocalNoopClustering>(ctx);

  case LocalClusteringAlgorithm::LP:
    return std::make_unique<LocalLPClustering>(ctx);
  }

  __builtin_unreachable();
}
} // namespace kaminpar::dist::factory
