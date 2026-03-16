/*******************************************************************************
 * Simple gain-based cluster selection strategy for label propagation.
 *
 * Used by the LP clusterer: selects the cluster with the highest gain that
 * satisfies the weight constraint. Does not consider overload balancing.
 *
 * @file:   simple_gain_selection.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/label_propagation/config.h"

namespace kaminpar::lp {

template <typename ClusterOps> class SimpleGainClusterSelection {
public:
  using ClusterID = decltype(std::declval<ClusterOps>().cluster(0));
  using ClusterWeight = decltype(std::declval<ClusterOps>().cluster_weight(ClusterID{}));
  using NodeWeight = ClusterWeight;
  using EdgeWeight = shm::EdgeWeight;

  using SelectionState = ClusterSelectionState<ClusterID, ClusterWeight, NodeWeight, EdgeWeight>;

  SimpleGainClusterSelection(ClusterOps &ops, shm::TieBreakingStrategy strategy)
      : _ops(ops),
        _tie_breaking_strategy(strategy) {}

  template <typename RatingMap>
  ClusterID select_best_cluster(
      const bool store_favored_cluster,
      const EdgeWeight gain_delta,
      SelectionState &state,
      RatingMap &map,
      std::vector<ClusterID> &tie_breaking_clusters,
      std::vector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const bool use_uniform_tie_breaking =
        _tie_breaking_strategy == shm::TieBreakingStrategy::UNIFORM;

    ClusterID favored_cluster = state.initial_cluster;
    if (use_uniform_tie_breaking) {
      const auto accept_cluster = [&] {
        return (state.current_cluster_weight + state.u_weight <=
                    _ops.max_cluster_weight(state.current_cluster) ||
                state.current_cluster == state.initial_cluster) &&
               _ops.accept_cluster(state.current_cluster, state.initial_cluster);
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = _ops.cluster_weight(cluster);

        if (store_favored_cluster) {
          if (state.current_gain > state.overall_best_gain) {
            state.overall_best_gain = state.current_gain;
            favored_cluster = state.current_cluster;

            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          } else if (state.current_gain == state.overall_best_gain) {
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          }
        }

        if (state.current_gain > state.best_gain) {
          if (accept_cluster()) {
            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(state.current_cluster);

            state.best_cluster = state.current_cluster;
            state.best_gain = state.current_gain;
          }
        } else if (state.current_gain == state.best_gain) {
          if (accept_cluster()) {
            tie_breaking_clusters.push_back(state.current_cluster);
          }
        }
      }

      if (tie_breaking_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_clusters.size());
        state.best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_favored_clusters.size());
        favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();

      return favored_cluster;
    } else {
      const auto accept_cluster = [&] {
        return (state.current_gain > state.best_gain ||
                (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
               (state.current_cluster_weight + state.u_weight <=
                    _ops.max_cluster_weight(state.current_cluster) ||
                state.current_cluster == state.initial_cluster) &&
               _ops.accept_cluster(state.current_cluster, state.initial_cluster);
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = _ops.cluster_weight(cluster);

        if (store_favored_cluster && state.current_gain > state.overall_best_gain) {
          state.overall_best_gain = state.current_gain;
          favored_cluster = state.current_cluster;
        }

        if (accept_cluster()) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      return favored_cluster;
    }
  }

private:
  ClusterOps &_ops;
  shm::TieBreakingStrategy _tie_breaking_strategy;
};

} // namespace kaminpar::lp
