/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   workspace.h
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/label_propagation/types.h"

namespace kaminpar::lp {

template <
    typename NodeID,
    typename ClusterID,
    typename EdgeWeight,
    typename RatingMap,
    typename GrowingRatingMap = DynamicRememberingFlatMap<ClusterID, EdgeWeight>,
    typename ConcurrentRatingMap = ConcurrentFastResetArray<EdgeWeight, ClusterID>,
    bool kEnableTwoPhase = true>
struct Workspace {
  static constexpr bool kSupportsTwoPhase = kEnableTwoPhase;

  using RatingMapType = RatingMap;
  using GrowingRatingMapType = GrowingRatingMap;
  using ConcurrentRatingMapType = ConcurrentRatingMap;

  tbb::enumerable_thread_specific<RatingMap> rating_map_ets;
  tbb::enumerable_thread_specific<GrowingRatingMap> growing_rating_map_ets;
  ConcurrentRatingMap concurrent_rating_map;
  tbb::enumerable_thread_specific<ScalableVector<ClusterID>> tie_breaking_clusters_ets;
  tbb::enumerable_thread_specific<ScalableVector<ClusterID>> tie_breaking_favored_clusters_ets;
  StaticArray<std::uint8_t> active;
  StaticArray<std::uint8_t> moved;
  StaticArray<ClusterID> favored_clusters;
  tbb::concurrent_vector<NodeID> second_phase_nodes;
  std::vector<LocalClusterSelectionState<ClusterID, EdgeWeight>> local_cluster_selection_states;

  void allocate(
      const NodeID num_nodes,
      const NodeID num_active_nodes,
      const ClusterID num_clusters,
      const ClusterID,
      const Options<NodeID, ClusterID> &options
  ) {
    if (options.active_set_strategy == ActiveSetStrategy::LOCAL) {
      if (active.size() < num_nodes) {
        active.resize(num_nodes);
      }
    } else if (options.active_set_strategy == ActiveSetStrategy::GLOBAL) {
      if (active.size() < num_active_nodes) {
        active.resize(num_active_nodes);
      }
    }

    if (options.use_two_hop_clustering) {
      if (favored_clusters.size() < num_active_nodes) {
        favored_clusters.resize(num_active_nodes);
      }
    }

    if (rating_map_ets.empty() || _rating_map_capacity < num_clusters) {
      rating_map_ets = tbb::enumerable_thread_specific<RatingMap>([num_clusters] {
        return RatingMap(num_clusters);
      });
      _rating_map_capacity = num_clusters;
    } else {
      for (auto &rating_map : rating_map_ets) {
        rating_map.change_max_size(num_clusters);
      }
    }

    if (local_cluster_selection_states.size() <
        static_cast<std::size_t>(tbb::this_task_arena::max_concurrency())) {
      local_cluster_selection_states.resize(tbb::this_task_arena::max_concurrency());
    }
  }

  void free() {
    rating_map_ets.clear();
    growing_rating_map_ets.clear();
    tie_breaking_clusters_ets.clear();
    tie_breaking_favored_clusters_ets.clear();
    active.free();
    moved.free();
    favored_clusters.free();
    second_phase_nodes.clear();
    second_phase_nodes.shrink_to_fit();
    local_cluster_selection_states.clear();
    concurrent_rating_map.free();
    _rating_map_capacity = 0;
  }

private:
  ClusterID _rating_map_capacity = 0;
};

} // namespace kaminpar::lp
