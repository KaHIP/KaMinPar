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
#include "kaminpar/refinement/label_propagation_refiner.h"
#include "kaminpar/refinement/multi_refiner.h"

namespace kaminpar::shm::factory {
std::unique_ptr<Coarsener>
create_coarsener(const Graph &graph, const CoarseningContext &c_ctx) {
  SCOPED_TIMER("Allocation");

  switch (c_ctx.algorithm) {
  case ClusteringAlgorithm::NOOP: {
    return std::make_unique<NoopCoarsener>();
  }

  case ClusteringAlgorithm::LABEL_PROPAGATION: {
    auto clustering_algorithm =
        std::make_unique<LPClustering>(graph.n(), c_ctx);
    auto coarsener = std::make_unique<ClusteringCoarsener>(
        std::move(clustering_algorithm), graph, c_ctx
    );
    return coarsener;
  }
  }

  __builtin_unreachable();
}

std::unique_ptr<ip::InitialRefiner> create_initial_refiner(
    const Graph &graph,
    const PartitionContext &p_ctx,
    const RefinementContext &r_ctx,
    ip::InitialRefiner::MemoryContext m_ctx
) {
  if (r_ctx.algorithms.empty()) {
    return std::make_unique<ip::InitialNoopRefiner>(std::move(m_ctx));
  }
  KASSERT(r_ctx.algorithms.size() == 1u,
          "multiple refinements during initial partitioning are not supported",
          assert::always);

  switch (r_ctx.algorithms.front()) {
  case RefinementAlgorithm::NOOP: {
    return std::make_unique<ip::InitialNoopRefiner>(std::move(m_ctx));
  }

  case RefinementAlgorithm::TWOWAY_FM: {
    switch (r_ctx.twoway_fm.stopping_rule) {
    case FMStoppingRule::SIMPLE:
      return std::make_unique<ip::InitialSimple2WayFM>(
          graph.n(), p_ctx, r_ctx, std::move(m_ctx)
      );
    case FMStoppingRule::ADAPTIVE:
      return std::make_unique<ip::InitialAdaptive2WayFM>(
          graph.n(), p_ctx, r_ctx, std::move(m_ctx)
      );
    }

    __builtin_unreachable();
  }

  case RefinementAlgorithm::LABEL_PROPAGATION:
  case RefinementAlgorithm::GREEDY_BALANCER:
  case RefinementAlgorithm::KWAY_FM:
    FATAL_ERROR << "Not implemented";
    return nullptr;
  }

  __builtin_unreachable();
}

namespace {
std::unique_ptr<Refiner>
create_refiner(const Context &ctx, const RefinementAlgorithm algorithm) {

  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return std::make_unique<NoopRefiner>();

  case RefinementAlgorithm::TWOWAY_FM:
    FATAL_ERROR << "Not implemented";
    return nullptr;

  case RefinementAlgorithm::LABEL_PROPAGATION:
    return std::make_unique<LabelPropagationRefiner>(ctx);

  case RefinementAlgorithm::GREEDY_BALANCER:
    return std::make_unique<GreedyBalancer>(ctx);

  case RefinementAlgorithm::KWAY_FM:
    return std::make_unique<FMRefiner>(ctx);
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
