/*******************************************************************************
 * @file:   factories.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Factory functions to instantiate coarsening and local improvement
 * algorithms.
 ******************************************************************************/
#include "kaminpar/factories.h"

#include <memory>

#include "kaminpar/coarsening/cluster_coarsener.h"
#include "kaminpar/coarsening/lp_clustering.h"
#include "kaminpar/coarsening/noop_coarsener.h"
#include "kaminpar/refinement/fm_refiner.h"
#include "kaminpar/refinement/greedy_balancer.h"
#include "kaminpar/refinement/jet_refiner.h"
#include "kaminpar/refinement/label_propagation_refiner.h"
#include "kaminpar/refinement/mtkahypar_refiner.h"
#include "kaminpar/refinement/multi_refiner.h"

namespace kaminpar::shm::factory {
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
    return std::make_unique<FMRefiner>(ctx);

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
