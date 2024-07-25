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
#include "kaminpar-shm/partitioning/kway/kway_multilevel.h"
#include "kaminpar-shm/partitioning/rb/rb_multilevel.h"

// Clusterings
#include "kaminpar-shm/coarsening/clustering/legacy_lp_clusterer.h"
#include "kaminpar-shm/coarsening/clustering/lp_clusterer.h"
#include "kaminpar-shm/coarsening/clustering/noop_clusterer.h"

// Coarsening
#include "kaminpar-shm/coarsening/cluster_coarsener.h"
#include "kaminpar-shm/coarsening/noop_coarsener.h"

// Refinement
#include <networkit/auxiliary/Multiprecision.hpp>
#include <networkit/sparsification/ForestFireScore.hpp>

#include "coarsening/sparsification/DensitySparsificationTarget.h"
#include "coarsening/sparsification/EdgeReductionSparsificationTarget.h"
#include "coarsening/sparsification/EffectiveResistanceScore.h"
#include "coarsening/sparsification/IndependentRandomSampler.h"
#include "coarsening/sparsification/NetworKitScoreAdapter.h"
#include "coarsening/sparsification/RandomWithReplacementSampler.h"
#include "coarsening/sparsification/RandomWithoutReplacementSampler.h"
#include "coarsening/sparsification/ThresholdSampler.h"
#include "coarsening/sparsification/UniformRandomSampler.h"
#include "coarsening/sparsification/WeightedForestFireScore.hpp"
#include "coarsening/sparsification/kNeighbourSampler.h"
#include "coarsening/sparsifing_cluster_coarsener.h"

#include "kaminpar-shm/refinement/adapters/mtkahypar_refiner.h"
#include "kaminpar-shm/refinement/balancer/greedy_balancer.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"
#include "kaminpar-shm/refinement/jet/jet_refiner.h"
#include "kaminpar-shm/refinement/lp/legacy_lp_refiner.h"
#include "kaminpar-shm/refinement/lp/lp_refiner.h"
#include "kaminpar-shm/refinement/multi_refiner.h"

namespace kaminpar::shm::factory {
std::unique_ptr<Partitioner> create_partitioner(const Graph &graph, const Context &ctx) {
  SCOPED_HEAP_PROFILER("Create partitioner");

  switch (ctx.partitioning.mode) {
  case PartitioningMode::DEEP:
    return std::make_unique<DeepMultilevelPartitioner>(graph, ctx);

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

  case ClusteringAlgorithm::LEGACY_LABEL_PROPAGATION:
    return std::make_unique<LegacyLPClustering>(ctx.coarsening);
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

  case CoarseningAlgorithm::CLUSTERING:
    return std::make_unique<ClusteringCoarsener>(ctx, p_ctx);

  case CoarseningAlgorithm::SPARSIFYING_CLUSTERING:
    return std::make_unique<SparsifyingClusteringCoarsener>(ctx, p_ctx);
  }

  __builtin_unreachable();
}

std::unique_ptr<sparsification::Sampler> create_sampler(const Context &ctx) {
  class WeightFunction : public sparsification::ScoreFunction<EdgeWeight> {
  public:
    StaticArray<EdgeWeight> scores(const CSRGraph &g) override {
      return StaticArray<EdgeWeight>(g.raw_edge_weights().begin(), g.raw_edge_weights().end());
    };
  };
  switch (ctx.sparsification.algorithm) {
  case SparsificationAlgorithm::FOREST_FIRE:
    return std::make_unique<sparsification::ThresholdSampler<double>>(
        std::make_unique<sparsification::NetworKitScoreAdapter<NetworKit::ForestFireScore, double>>(
            [](const NetworKit::Graph &g) { return NetworKit::ForestFireScore(g, 0.95, 5); }
        ),
        std::make_unique<sparsification::IdentityReweihingFunction<double>>()
    );
  case SparsificationAlgorithm::UNIFORM_RANDOM_SAMPLING:
    return std::make_unique<sparsification::UniformRandomSampler>();
  case SparsificationAlgorithm::K_NEIGHBOUR:
    return std::make_unique<sparsification::kNeighbourSampler>();
  case SparsificationAlgorithm::K_NEIGHBOUR_SPANNING_TREE:
    return std::make_unique<sparsification::kNeighbourSampler>(true);
  case SparsificationAlgorithm::WEIGHT_THRESHOLD:
    return std::make_unique<sparsification::ThresholdSampler<EdgeWeight>>(
        std::make_unique<WeightFunction>(),
        std::make_unique<sparsification::IdentityReweihingFunction<EdgeWeight>>()
    );
  case SparsificationAlgorithm::EFFECTIVE_RESISTANCE:
    return std::make_unique<sparsification::ThresholdSampler<double>>(
        std::make_unique<sparsification::EffectiveResistanceScore>(4),
        std::make_unique<sparsification::IdentityReweihingFunction<double>>()
    );
  case SparsificationAlgorithm::RANDOM_WITH_REPLACEMENT:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::RandomWithReplacementSampler<double>>(
          std::make_unique<
              sparsification::NetworKitScoreAdapter<NetworKit::ForestFireScore, double>>(
              [](const NetworKit::Graph &g) { return NetworKit::ForestFireScore(g, 0.95, 5); }
          ),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::RandomWithReplacementSampler<double>>(
          std::make_unique<sparsification::NetworKitScoreAdapter<
              sparsification::WeightedForestFireScore,
              double>>([](const NetworKit::Graph &g) {
            return sparsification::WeightedForestFireScore(g, 0.95, 5);
          }),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::EFFECTIVE_RESISTANCE:
      return std::make_unique<sparsification::RandomWithReplacementSampler<double>>(
          std::make_unique<sparsification::EffectiveResistanceScore>(4),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::RandomWithReplacementSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>(),
          std::make_unique<sparsification::IdentityReweihingFunction<EdgeWeight>>()
      );
    }
  case SparsificationAlgorithm::RANDOM_WITHOUT_REPLACEMENT:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::RandomWithoutReplacementSampler<double>>(
          std::make_unique<
              sparsification::NetworKitScoreAdapter<NetworKit::ForestFireScore, double>>(
              [](const NetworKit::Graph &g) { return NetworKit::ForestFireScore(g, 0.95, 5); }
          ),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::RandomWithoutReplacementSampler<double>>(
          std::make_unique<sparsification::NetworKitScoreAdapter<
              sparsification::WeightedForestFireScore,
              double>>([](const NetworKit::Graph &g) {
            return sparsification::WeightedForestFireScore(g, 0.95, 5);
          }),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::EFFECTIVE_RESISTANCE:
      return std::make_unique<sparsification::RandomWithoutReplacementSampler<double>>(
          std::make_unique<sparsification::EffectiveResistanceScore>(4),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::RandomWithoutReplacementSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>(),
          std::make_unique<sparsification::IdentityReweihingFunction<EdgeWeight>>()
      );
    }
  case SparsificationAlgorithm::INDEPENDENT_RANDOM:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::IndependentRandomSampler<double>>(
          std::make_unique<
              sparsification::NetworKitScoreAdapter<NetworKit::ForestFireScore, double>>(
              [](const NetworKit::Graph &g) { return NetworKit::ForestFireScore(g, 0.95, 5); }
          ),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::IndependentRandomSampler<double>>(
          std::make_unique<sparsification::NetworKitScoreAdapter<
              sparsification::WeightedForestFireScore,
              double>>([](const NetworKit::Graph &g) {
            return sparsification::WeightedForestFireScore(g, 0.95, 5);
          }),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::EFFECTIVE_RESISTANCE:
      return std::make_unique<sparsification::IndependentRandomSampler<double>>(
          std::make_unique<sparsification::EffectiveResistanceScore>(4),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::IndependentRandomSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>(),
          std::make_unique<sparsification::IdentityReweihingFunction<EdgeWeight>>()
      );
    }
  case SparsificationAlgorithm::THRESHOLD:
    switch (ctx.sparsification.score_function) {
    case ScoreFunctionSection::FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<double>>(
          std::make_unique<
              sparsification::NetworKitScoreAdapter<NetworKit::ForestFireScore, double>>(
              [](const NetworKit::Graph &g) { return NetworKit::ForestFireScore(g, 0.95, 5); }
          ),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHTED_FOREST_FIRE:
      return std::make_unique<sparsification::ThresholdSampler<double>>(
          std::make_unique<sparsification::NetworKitScoreAdapter<
              sparsification::WeightedForestFireScore,
              double>>([](const NetworKit::Graph &g) {
            return sparsification::WeightedForestFireScore(g, 0.95, 5);
          }),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::EFFECTIVE_RESISTANCE:
      return std::make_unique<sparsification::ThresholdSampler<double>>(
          std::make_unique<sparsification::EffectiveResistanceScore>(4),
          std::make_unique<sparsification::IdentityReweihingFunction<double>>()
      );
    case ScoreFunctionSection::WEIGHT:
      return std::make_unique<sparsification::ThresholdSampler<EdgeWeight>>(
          std::make_unique<WeightFunction>(),
          std::make_unique<sparsification::IdentityReweihingFunction<EdgeWeight>>()
      );
    }
  }

  __builtin_unreachable();
}

std::unique_ptr<sparsification::SparsificationTarget>
create_sparsification_target(const Context &ctx) {
  switch (ctx.sparsification.target) {
  case SparsificationTargetSelection::DENSITY:
    return std::make_unique<sparsification::DensitySparsificationTarget>(
        ctx.sparsification.target_factor
    );
  case SparsificationTargetSelection::EDGE_REDUCTION:
    return std::make_unique<sparsification::EdgeReductionSparsificationTarget>(
        ctx.sparsification.target_factor
    );
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

  case RefinementAlgorithm::LEGACY_LABEL_PROPAGATION:
    return std::make_unique<LegacyLabelPropagationRefiner>(ctx);

  case RefinementAlgorithm::GREEDY_BALANCER:
    return std::make_unique<GreedyBalancer>(ctx);

  case RefinementAlgorithm::KWAY_FM:
    return create_fm_refiner(ctx);

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
