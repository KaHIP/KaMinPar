/*******************************************************************************
 * Overload-aware cluster selection strategy for label propagation.
 *
 * Used by the LP refiner: selects the cluster with the highest gain, but also
 * considers block overload (imbalance). Nodes may move to overloaded blocks if
 * doing so reduces overall imbalance, and nodes cannot leave blocks that would
 * fall below the minimum weight constraint.
 *
 * @file:   overload_aware_selection.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include <type_traits>
#include <vector>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/label_propagation/config.h"

namespace kaminpar::lp {

template <typename ClusterOps> class OverloadAwareClusterSelection {
public:
  using ClusterID = decltype(std::declval<ClusterOps>().cluster(0));
  using ClusterWeight = decltype(std::declval<ClusterOps>().cluster_weight(ClusterID{}));
  using NodeWeight = ClusterWeight;
  using EdgeWeight = shm::EdgeWeight;

  using SelectionState = ClusterSelectionState<ClusterID, ClusterWeight, NodeWeight, EdgeWeight>;

  OverloadAwareClusterSelection(ClusterOps &ops, shm::TieBreakingStrategy strategy)
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
    // Don't move node if it would make the source block underweight.
    if (state.initial_cluster_weight - state.u_weight < _ops.min_cluster_weight(state.initial_cluster)) {
      return state.initial_cluster;
    }

    const bool use_uniform_tie_breaking =
        _tie_breaking_strategy == shm::TieBreakingStrategy::UNIFORM;

    ClusterID favored_cluster = state.initial_cluster;
    if (use_uniform_tie_breaking) {
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
          const NodeWeight current_max_weight = _ops.max_cluster_weight(state.current_cluster);
          const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
          const NodeWeight initial_overload =
              state.initial_cluster_weight - _ops.max_cluster_weight(state.initial_cluster);

          if (((state.current_cluster_weight + state.u_weight <= current_max_weight)) ||
              current_overload < initial_overload ||
              state.current_cluster == state.initial_cluster) {
            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(state.current_cluster);

            state.best_cluster = state.current_cluster;
            state.best_cluster_weight = state.current_cluster_weight;
            state.best_gain = state.current_gain;
          }
        } else if (state.current_gain == state.best_gain) {
          const NodeWeight current_max_weight = _ops.max_cluster_weight(state.current_cluster);
          const NodeWeight best_overload =
              state.best_cluster_weight - _ops.max_cluster_weight(state.best_cluster);
          const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;

          if (current_overload < best_overload) {
            const NodeWeight initial_overload =
                state.initial_cluster_weight - _ops.max_cluster_weight(state.initial_cluster);

            if (((state.current_cluster_weight + state.u_weight <= current_max_weight)) ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster) {
              tie_breaking_clusters.clear();
              tie_breaking_clusters.push_back(state.current_cluster);

              state.best_cluster = state.current_cluster;
              state.best_cluster_weight = state.current_cluster_weight;
            }
          } else if (current_overload == best_overload) {
            const NodeWeight initial_overload =
                state.initial_cluster_weight - _ops.max_cluster_weight(state.initial_cluster);

            if (state.current_cluster_weight + state.u_weight <= current_max_weight ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster) {
              tie_breaking_clusters.push_back(state.current_cluster);
            }
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
      static_assert(std::is_signed_v<NodeWeight>);

      const auto accept_cluster = [&] {
        const NodeWeight current_max_weight = _ops.max_cluster_weight(state.current_cluster);
        const NodeWeight best_overload =
            state.best_cluster_weight - _ops.max_cluster_weight(state.best_cluster);
        const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
        const NodeWeight initial_overload =
            state.initial_cluster_weight - _ops.max_cluster_weight(state.initial_cluster);

        return (state.current_gain > state.best_gain ||
                (state.current_gain == state.best_gain &&
                 (current_overload < best_overload ||
                  (current_overload == best_overload && state.local_rand.random_bool())))) &&
               (((state.current_cluster_weight + state.u_weight <= current_max_weight)) ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster);
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
