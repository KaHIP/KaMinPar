/*******************************************************************************
 * Factory functions to instantiate partitioning composed based on their
 * respective enum constant.
 *
 * @file:   factories.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/factories.h"

#include <memory>

// Partitioning schemes
#include "kaminpar-shm/partitioning/deep/deep_multilevel.h"
#include "kaminpar-shm/partitioning/deep/vcycle_deep_multilevel.h"
#include "kaminpar-shm/partitioning/kway/kway_multilevel.h"
#include "kaminpar-shm/partitioning/rb/rb_multilevel.h"

// Clusterings
#include "kaminpar-shm/coarsening/clustering/hem_clusterer.h"
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"
#include "kaminpar-shm/coarsening/clustering/noop_clusterer.h"

// Coarsening
#include "kaminpar-shm/coarsening/basic_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/noop_coarsener.h"
#include "kaminpar-shm/coarsening/overlay_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/sparsification_cluster_coarsener.h"

// Refinement
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"
#include "kaminpar-shm/refinement/balancer/underload_balancer.h"
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"
#include "kaminpar-shm/refinement/fm/unconstrained_fm_refiner.h"
#include "kaminpar-shm/refinement/jet/jet_refiner.h"
#include "kaminpar-shm/refinement/lp/lp_refiner.h"
#include "kaminpar-shm/refinement/lp/unconstrained_lp_refiner.h"
#include "kaminpar-shm/refinement/multi_refiner.h"

namespace kaminpar::shm::factory {

std::unique_ptr<Partitioner> create_partitioner(const Graph &graph, const ExecutionContext &ctx) {
  SCOPED_HEAP_PROFILER("Create partitioner");

  switch (ctx.config.partitioning.mode) {
  case PartitioningMode::DEEP:
    return std::make_unique<DeepMultilevelPartitioner>(graph, ctx);

  case PartitioningMode::VCYCLE:
    return std::make_unique<VcycleDeepMultilevelPartitioner>(graph, ctx);

  case PartitioningMode::RB:
    return std::make_unique<RBMultilevelPartitioner>(graph, ctx);

  case PartitioningMode::KWAY:
    return std::make_unique<KWayMultilevelPartitioner>(graph, ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<Clusterer> create_clusterer(const ExecutionContext &ctx) {
  switch (ctx.config.coarsening.clustering.algorithm) {
  case ClusteringAlgorithm::NOOP:
    return std::make_unique<NoopClusterer>();

  case ClusteringAlgorithm::LABEL_PROPAGATION:
    return std::make_unique<LPClustering>(ctx.config.coarsening, ctx.runtime);

  case ClusteringAlgorithm::HEAVY_EDGE_MATCHING:
    return std::make_unique<HEMClustering>(ctx.config.coarsening, ctx.runtime.timer);
  }

  __builtin_unreachable();
}

std::unique_ptr<Coarsener> create_coarsener(const ExecutionContext &ctx) {
  return create_coarsener(ctx, ctx.partition);
}

std::unique_ptr<Coarsener>
create_coarsener(const ExecutionContext &ctx, const PartitionContext &p_ctx) {
  switch (ctx.config.coarsening.algorithm) {
  case CoarseningAlgorithm::NOOP:
    return std::make_unique<NoopCoarsener>();

  case CoarseningAlgorithm::BASIC_CLUSTERING:
    return std::make_unique<BasicClusterCoarsener>(ctx, p_ctx);

  case CoarseningAlgorithm::OVERLAY_CLUSTERING:
    return std::make_unique<OverlayClusterCoarsener>(ctx, p_ctx);

  case CoarseningAlgorithm::SPARSIFICATION_CLUSTERING:
    return std::make_unique<SparsificationClusterCoarsener>(ctx, p_ctx);
  }

  __builtin_unreachable();
}

namespace {

std::unique_ptr<Refiner>
create_refiner(const ExecutionContext &ctx, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return std::make_unique<NoopRefiner>();

  case RefinementAlgorithm::OVERLOAD_BALANCER:
    return std::make_unique<BlockParallelOverloadBalancer>(ctx);

  case RefinementAlgorithm::UNDERLOAD_BALANCER:
    return std::make_unique<UnderloadBalancer>(ctx);

  case RefinementAlgorithm::LABEL_PROPAGATION:
    return std::make_unique<LabelPropagationRefiner>(ctx);

  case RefinementAlgorithm::UNCONSTRAINED_LABEL_PROPAGATION:
    return std::make_unique<UnconstrainedLabelPropagationRefiner>(ctx);

  case RefinementAlgorithm::KWAY_FM:
    return std::make_unique<FMRefiner>(ctx);

  case RefinementAlgorithm::UNCONSTRAINED_FM:
    return std::make_unique<UnconstrainedFMRefiner>(ctx);

  case RefinementAlgorithm::TWOWAY_FLOW:
    return std::make_unique<TwowayFlowRefiner>(
        ctx.parallel, ctx.config.refinement.twoway_flow, ctx.runtime
    );

  case RefinementAlgorithm::JET:
    return std::make_unique<JetRefiner>(ctx);
  }

  __builtin_unreachable();
}

} // namespace

std::unique_ptr<Refiner> create_refiner(const ExecutionContext &ctx) {
  SCOPED_HEAP_PROFILER("Refiner Allocation");

  if (ctx.config.refinement.algorithms.empty()) {
    return std::make_unique<NoopRefiner>();
  }
  if (ctx.config.refinement.algorithms.size() == 1) {
    return create_refiner(ctx, ctx.config.refinement.algorithms.front());
  }

  std::unordered_map<RefinementAlgorithm, std::unique_ptr<Refiner>> refiners;
  for (const RefinementAlgorithm algorithm : ctx.config.refinement.algorithms) {
    if (refiners.find(algorithm) == refiners.end()) {
      refiners[algorithm] = create_refiner(ctx, algorithm);
    }
  }

  return std::make_unique<MultiRefiner>(
      std::move(refiners), ctx.config.refinement.algorithms, ctx.runtime
  );
}

} // namespace kaminpar::shm::factory
