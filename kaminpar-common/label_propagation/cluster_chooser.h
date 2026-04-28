/*******************************************************************************
 * Shared best-cluster selection loop.
 *
 * @file:   cluster_chooser.h
 ******************************************************************************/
#pragma once

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/label_propagation/types.h"

namespace kaminpar::lp {

template <typename EdgeWeight>
[[nodiscard]] KAMINPAR_LP_INLINE CandidateComparison
compare_by_gain(const EdgeWeight candidate_gain, const EdgeWeight best_gain) {
  if (candidate_gain > best_gain) {
    return CandidateComparison::BETTER;
  }
  if (candidate_gain == best_gain) {
    return CandidateComparison::EQUIVALENT;
  }
  return CandidateComparison::WORSE;
}

template <typename Context>
[[nodiscard]] KAMINPAR_LP_INLINE auto make_initial_choice(const Context &context) {
  using ClusterID = typename Context::ClusterIDType;
  using ClusterWeight = typename Context::ClusterWeightType;
  using EdgeWeight = typename Context::EdgeWeightType;

  return ClusterChoice<ClusterID, ClusterWeight, EdgeWeight>{
      .best_cluster = context.initial_cluster,
      .best_gain = 0,
      .best_cluster_weight = context.initial_cluster_weight,
      .favored_cluster = context.initial_cluster,
      .favored_gain = 0,
  };
}

template <typename Choice, typename Candidate>
KAMINPAR_LP_INLINE void set_best_cluster(Choice &choice, const Candidate &candidate) {
  choice.best_cluster = candidate.cluster;
  choice.best_gain = candidate.gain;
  choice.best_cluster_weight = candidate.weight;
}

template <typename Choice, typename Candidate>
KAMINPAR_LP_INLINE void set_favored_cluster(Choice &choice, const Candidate &candidate) {
  choice.favored_cluster = candidate.cluster;
  choice.favored_gain = candidate.gain;
}

template <TieBreakingStrategy Strategy, typename Context, typename Evaluator, typename RatingMap>
[[nodiscard]] KAMINPAR_LP_INLINE auto choose_cluster(
    const Context &context,
    RatingMap &map,
    Evaluator &evaluator,
    ScalableVector<typename Context::ClusterIDType> &tie_breaking_clusters,
    ScalableVector<typename Context::ClusterIDType> &tie_breaking_favored_clusters
) {
  using ClusterID = typename Context::ClusterIDType;
  using ClusterWeight = typename Context::ClusterWeightType;
  using EdgeWeight = typename Context::EdgeWeightType;
  using Candidate = ClusterCandidate<ClusterID, ClusterWeight, EdgeWeight>;

  auto choice = make_initial_choice(context);

  for (const auto [cluster, rating] : map.entries()) {
    const Candidate candidate{
        .cluster = static_cast<ClusterID>(cluster),
        .gain = rating - context.gain_delta,
        .weight = evaluator.cluster_weight(cluster),
    };

    if (context.track_favored_cluster) {
      const CandidateComparison favored_comparison =
          compare_by_gain(candidate.gain, choice.favored_gain);
      if constexpr (Strategy == TieBreakingStrategy::UNIFORM) {
        if (favored_comparison == CandidateComparison::BETTER) {
          set_favored_cluster(choice, candidate);
          tie_breaking_favored_clusters.clear();
          tie_breaking_favored_clusters.push_back(candidate.cluster);
        } else if (favored_comparison == CandidateComparison::EQUIVALENT) {
          tie_breaking_favored_clusters.push_back(candidate.cluster);
        }
      } else {
        if (favored_comparison == CandidateComparison::BETTER) {
          set_favored_cluster(choice, candidate);
        }
      }
    }

    const CandidateComparison comparison = evaluator.compare(context, candidate, choice);
    if constexpr (Strategy == TieBreakingStrategy::UNIFORM) {
      if (comparison == CandidateComparison::BETTER) {
        if (evaluator.is_feasible(context, candidate, choice)) {
          set_best_cluster(choice, candidate);
          tie_breaking_clusters.clear();
          tie_breaking_clusters.push_back(candidate.cluster);
        }
      } else if (comparison == CandidateComparison::EQUIVALENT) {
        if (evaluator.is_feasible(context, candidate, choice)) {
          tie_breaking_clusters.push_back(candidate.cluster);
        }
      }
    } else {
      if (comparison == CandidateComparison::BETTER ||
          (comparison == CandidateComparison::EQUIVALENT && context.rand.random_bool())) {
        if (evaluator.is_feasible(context, candidate, choice)) {
          set_best_cluster(choice, candidate);
        }
      }
    }
  }

  if constexpr (Strategy == TieBreakingStrategy::UNIFORM) {
    if (tie_breaking_clusters.size() > 1) {
      const ClusterID i = context.rand.random_index(0, tie_breaking_clusters.size());
      choice.best_cluster = tie_breaking_clusters[i];
    }
    tie_breaking_clusters.clear();

    if (tie_breaking_favored_clusters.size() > 1) {
      const ClusterID i = context.rand.random_index(0, tie_breaking_favored_clusters.size());
      choice.favored_cluster = tie_breaking_favored_clusters[i];
    }
    tie_breaking_favored_clusters.clear();
  }

  if constexpr (requires { evaluator.record_choice(context, choice); }) {
    evaluator.record_choice(context, choice);
  }

  return choice;
}

template <typename Context, typename Evaluator, typename RatingMap>
[[nodiscard]] KAMINPAR_LP_INLINE auto choose_cluster(
    const Context &context,
    RatingMap &map,
    Evaluator &evaluator,
    const TieBreakingStrategy tie_breaking,
    ScalableVector<typename Context::ClusterIDType> &tie_breaking_clusters,
    ScalableVector<typename Context::ClusterIDType> &tie_breaking_favored_clusters
) {
  switch (tie_breaking) {
  case TieBreakingStrategy::GEOMETRIC:
    return choose_cluster<TieBreakingStrategy::GEOMETRIC>(
        context, map, evaluator, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  case TieBreakingStrategy::UNIFORM:
    return choose_cluster<TieBreakingStrategy::UNIFORM>(
        context, map, evaluator, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }
  __builtin_unreachable();
}

} // namespace kaminpar::lp
