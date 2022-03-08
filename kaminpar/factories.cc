/*******************************************************************************
 * @file:   factories.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Factory functions to instantiate coarsening and local improvement
 * algorithms.
 ******************************************************************************/
#include "kaminpar/factories.h"

#include "kaminpar/coarsening/cluster_coarsener.h"
#include "kaminpar/coarsening/label_propagation_clustering.h"
#include "kaminpar/coarsening/noop_coarsener.h"
#include "kaminpar/refinement/greedy_balancer.h"
#include "kaminpar/refinement/label_propagation_refiner.h"

#include <memory>

namespace kaminpar::factory {
std::unique_ptr<ICoarsener> create_coarsener(const Graph &graph, const CoarseningContext &c_ctx) {
  SCOPED_TIMER("Allocation");

  switch (c_ctx.algorithm) {
  case ClusteringAlgorithm::NOOP: {
    return std::make_unique<NoopCoarsener>();
  }

  case ClusteringAlgorithm::LABEL_PROPAGATION: {
    auto clustering_algorithm = std::make_unique<LabelPropagationClusteringAlgorithm>(graph.n(), c_ctx);
    auto coarsener = std::make_unique<ClusteringCoarsener>(std::move(clustering_algorithm), graph, c_ctx);
    return coarsener;
  }
  }

  __builtin_unreachable();
}

std::unique_ptr<ip::InitialRefiner> create_initial_refiner(const Graph &graph, const PartitionContext &p_ctx,
                                                           const RefinementContext &r_ctx,
                                                           ip::InitialRefiner::MemoryContext m_ctx) {
  switch (r_ctx.algorithm) {
  case RefinementAlgorithm::NOOP: {
    return std::make_unique<ip::InitialNoopRefiner>(std::move(m_ctx));
  }

  case RefinementAlgorithm::TWO_WAY_FM: {
    switch (r_ctx.fm.stopping_rule) {
    case FMStoppingRule::SIMPLE:
      return std::make_unique<ip::InitialSimple2WayFM>(graph.n(), p_ctx, r_ctx, std::move(m_ctx));
    case FMStoppingRule::ADAPTIVE:
      return std::make_unique<ip::InitialAdaptive2WayFM>(graph.n(), p_ctx, r_ctx, std::move(m_ctx));
    }

    __builtin_unreachable();
  }

  case RefinementAlgorithm::LABEL_PROPAGATION: {
    FATAL_ERROR << "Not implemented";
    return nullptr;
  }
  }

  __builtin_unreachable();
}

std::unique_ptr<IRefiner> create_refiner(const Context &ctx) {
  SCOPED_TIMER("Allocation");

  switch (ctx.refinement.algorithm) {
  case RefinementAlgorithm::NOOP: {
    return std::make_unique<NoopRefiner>();
  }

  case RefinementAlgorithm::TWO_WAY_FM: {
    FATAL_ERROR << "Not implemented";
    return nullptr;
  }

  case RefinementAlgorithm::LABEL_PROPAGATION: {
    return std::make_unique<LabelPropagationRefiner>(ctx);
  }
  }

  __builtin_unreachable();
}

std::unique_ptr<IBalancer> create_balancer(const Graph &graph, const PartitionContext &p_ctx,
                                           const RefinementContext &r_ctx) {
  SCOPED_TIMER("Allocation");

  switch (r_ctx.balancer.algorithm) {
  case BalancingAlgorithm::NOOP: {
    return std::make_unique<NoopBalancer>();
  }

  case BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER: {
    return std::make_unique<GreedyBalancer>(graph, p_ctx.k, r_ctx);
  }
  }

  __builtin_unreachable();
}
} // namespace kaminpar::factory