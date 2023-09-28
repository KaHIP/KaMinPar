/*******************************************************************************
 * Factory functions to instantiate partitioning composed based on their
 * respective enum constant.
 *
 * @file:   factories.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar/factories.h"

#include <memory>

// Partitioning schemes
#include "kaminpar/partitioning/deep/deep_multilevel.h"
#include "kaminpar/partitioning/kway/kway_multilevel.h"
#include "kaminpar/partitioning/rb/rb_multilevel.h"

// Clusterings
#include "kaminpar/coarsening/cluster_coarsener.h"
#include "kaminpar/coarsening/lp_clustering.h"

// Coarsening
#include "kaminpar/coarsening/noop_coarsener.h"

// Refinement
#include "kaminpar/refinement/balancer/greedy_balancer.h"
#include "kaminpar/refinement/fm/fm_refiner.h"
#include "kaminpar/refinement/jet/jet_refiner.h"
#include "kaminpar/refinement/lp/lp_refiner.h"
#include "kaminpar/refinement/mtkahypar_refiner.h"
#include "kaminpar/refinement/multi_refiner.h"

namespace kaminpar::shm::factory {
std::unique_ptr<Partitioner> create_partitioner(const Graph &graph, const Context &ctx) {
  switch (ctx.partitioning.mode) {
  case PartitioningMode::DEEP: {
    return std::make_unique<DeepMultilevelPartitioner>(graph, ctx);
  }

  case PartitioningMode::RB: {
    return std::make_unique<RBMultilevelPartitioner>(graph, ctx);
  }

  case PartitioningMode::KWAY: {
    return std::make_unique<KWayMultilevelPartitioner>(graph, ctx);
  }
  }

  __builtin_unreachable();
}

std::unique_ptr<Coarsener> create_coarsener(const Graph &graph, const CoarseningContext &c_ctx) {
  SCOPED_TIMER("Allocation");

  switch (c_ctx.algorithm) {
  case ClusteringAlgorithm::NOOP: {
    return std::make_unique<NoopCoarsener>();
  }

  case ClusteringAlgorithm::LABEL_PROPAGATION: {
    auto clustering_algorithm = std::make_unique<LPClustering>(graph.n(), c_ctx);
    auto coarsener =
        std::make_unique<ClusteringCoarsener>(std::move(clustering_algorithm), graph, c_ctx);
    return coarsener;
  }
  }

  __builtin_unreachable();
}

namespace {
std::unique_ptr<Refiner> create_refiner(const Context &ctx, const RefinementAlgorithm algorithm) {

  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return std::make_unique<NoopRefiner>();

  case RefinementAlgorithm::LABEL_PROPAGATION:
    return std::make_unique<LabelPropagationRefiner>(ctx);

  case RefinementAlgorithm::GREEDY_BALANCER:
    return std::make_unique<GreedyBalancer>(ctx);

  case RefinementAlgorithm::KWAY_FM:
    return std::make_unique<FMRefiner<>>(ctx);

  case RefinementAlgorithm::JET:
    return std::make_unique<JetRefiner>(ctx);

  case RefinementAlgorithm::MTKAHYPAR:
    return std::make_unique<MtKaHyParRefiner>(ctx);
  }

  __builtin_unreachable();
}
} // namespace

std::unique_ptr<Refiner> create_refiner(const Context &ctx) {
  SCOPED_TIMER("Allocation");

  if (ctx.refinement.algorithms.empty()) {
    return std::make_unique<NoopRefiner>();
  }
  if (ctx.refinement.algorithms.size() == 1) {
    return create_refiner(ctx, ctx.refinement.algorithms.front());
  }

  std::vector<std::unique_ptr<Refiner>> refiners;
  for (const RefinementAlgorithm algorithm : ctx.refinement.algorithms) {
    refiners.push_back(create_refiner(ctx, algorithm));
  }
  return std::make_unique<MultiRefiner>(std::move(refiners));
}
} // namespace kaminpar::shm::factory
