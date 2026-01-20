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
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"
#include "kaminpar-shm/coarsening/clustering/noop_clusterer.h"

// Coarsening
#include "kaminpar-shm/coarsening/basic_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/noop_coarsener.h"
#include "kaminpar-shm/coarsening/overlay_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/sparsifying_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/threshold_sparsifying_cluster_coarsener.h"

// Sparsification
#include "kaminpar-shm/coarsening/sparsification/independent_random_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/k_neighbor_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/random_with_replacement_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/random_without_replacement_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/threshold_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/unbiased_threshold_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/uniform_random_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/weighted_forest_fire_score.h"

// Refinement
#include "kaminpar-shm/refinement/adapters/mtkahypar_refiner.h"
#include "kaminpar-shm/refinement/balancer/greedy_balancer.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"
#include "kaminpar-shm/refinement/jet/jet_refiner.h"
#include "kaminpar-shm/refinement/lp/lp_refiner.h"
#include "kaminpar-shm/refinement/multi_refiner.h"

namespace kaminpar::shm::factory {

std::unique_ptr<Partitioner> create_partitioner(const Graph &graph, const Context &ctx) {
  SCOPED_HEAP_PROFILER("Create partitioner");

  switch (ctx.partitioning.mode) {
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

std::unique_ptr<Clusterer> create_clusterer(const Context &ctx) {
  switch (ctx.coarsening.clustering.algorithm) {
  case ClusteringAlgorithm::NOOP:
    return std::make_unique<NoopClusterer>();

  case ClusteringAlgorithm::LABEL_PROPAGATION:
    return std::make_unique<LPClustering>(ctx.coarsening);
  }

  __builtin_unreachable();
}

std::unique_ptr<Coarsener> create_coarsener(const Context &ctx) {
  return create_coarsener(ctx, ctx.partition);
}

std::unique_ptr<Coarsener> create_coarsener(const Context &ctx, const PartitionContext &p_ctx) {
  switch (ctx.coarsening.algorithm) {
  case CoarseningAlgorithm::NOOP:
    return std::make_unique<NoopCoarsener>();

  case CoarseningAlgorithm::BASIC_CLUSTERING:
    return std::make_unique<BasicClusterCoarsener>(ctx, p_ctx);

  case CoarseningAlgorithm::OVERLAY_CLUSTERING:
    return std::make_unique<OverlayClusterCoarsener>(ctx, p_ctx);

  case CoarseningAlgorithm::SPARSIFYING_CLUSTERING:
    return std::make_unique<SparsifyingClusterCoarsener>(ctx, p_ctx);

  case CoarseningAlgorithm::THRESHOLD_SPARSIFYING_CLUSTERING:
    return std::make_unique<ThresholdSparsifyingClusterCoarsener>(ctx, p_ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<sparsification::Sampler> create_sampler(const Context &ctx) {
  class WeightFunction : public sparsification::ScoreFunction<EdgeWeight> {
  public:
    StaticArray<EdgeWeight> scores(const CSRGraph &g) override {
      return StaticArray<EdgeWeight>(g.raw_edge_weights().begin(), g.raw_edge_weights().end());
    }
  };

  switch (ctx.sparsification.algorithm) {
  case SparsificationAlgorithm::UNIFORM_RANDOM_SAMPLING:
    return std::make_unique<sparsification::UniformRandomSampler>();

  case SparsificationAlgorithm::WEIGHTED_UNIFORM_RANDOM_SAMPLING:
    return std::make_unique<sparsification::WeightedUniformRandomSampler>();

  case SparsificationAlgorithm::K_NEIGHBOUR:
    return std::make_unique<sparsification::kNeighborSampler>();

  case SparsificationAlgorithm::K_NEIGHBOUR_SPANNING_TREE:
    return std::make_unique<sparsification::kNeighborSampler>(true);

  case SparsificationAlgorithm::UNBIASED_THRESHOLD:
    return std::make_unique<sparsification::UnbiasedThesholdSampler>();

  case SparsificationAlgorithm::WEIGHT_THRESHOLD:
    return std::make_unique<sparsification::ThresholdSampler<EdgeWeight>>(
        std::make_unique<WeightFunction>()
    );

  case SparsificationAlgorithm::RANDOM_WITH_REPLACEMENT:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::RandomWithReplacementSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio
          )
      );

    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio, true
          )
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::RandomWithReplacementSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>()

      );
    }
    break;

  case SparsificationAlgorithm::RANDOM_WITHOUT_REPLACEMENT:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::RandomWithoutReplacementSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio
          )
      );

    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio, true
          )
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::RandomWithoutReplacementSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>()

      );
    }
    break;

  case SparsificationAlgorithm::INDEPENDENT_RANDOM:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::IndependentRandomSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio
          ),
          ctx.sparsification.no_approx
      );

    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio, true
          )
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::IndependentRandomSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>(), ctx.sparsification.no_approx
      );
    }
    break;

  case SparsificationAlgorithm::THRESHOLD:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio
          )
      );

    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<EdgeID>>(
          std::make_unique<sparsification::WeightedForestFireScore>(
              ctx.sparsification.wff_pf, ctx.sparsification.wff_target_burnt_ratio, true
          )
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::ThresholdSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>()

      );
    }
    break;
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
  SCOPED_HEAP_PROFILER("Refiner Allocation");
  SCOPED_TIMER("Refinement");
  SCOPED_TIMER("Allocation");

  if (ctx.refinement.algorithms.empty()) {
    return std::make_unique<NoopRefiner>();
  }
  if (ctx.refinement.algorithms.size() == 1) {
    return create_refiner(ctx, ctx.refinement.algorithms.front());
  }

  std::unordered_map<RefinementAlgorithm, std::unique_ptr<Refiner>> refiners;
  for (const RefinementAlgorithm algorithm : ctx.refinement.algorithms) {
    if (refiners.find(algorithm) == refiners.end()) {
      refiners[algorithm] = create_refiner(ctx, algorithm);
    }
  }

  return std::make_unique<MultiRefiner>(std::move(refiners), ctx.refinement.algorithms);
}

} // namespace kaminpar::shm::factory
