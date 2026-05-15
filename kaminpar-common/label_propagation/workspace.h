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

#include "kaminpar-common/datastructures/cache_aligned_vector.h"
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

  struct RatingBuffers {
    tbb::enumerable_thread_specific<RatingMap> maps;
    tbb::enumerable_thread_specific<GrowingRatingMap> growing_maps;
    ClusterID map_capacity = 0;

    void free() {
      maps.clear();
      growing_maps.clear();
      map_capacity = 0;
    }
  };

  struct ActiveSetStorage {
    StaticArray<std::uint8_t> flags;

    void free() {
      flags.free();
    }
  };

  struct SelectionBuffers {
    tbb::enumerable_thread_specific<ScalableVector<ClusterID>> tie_breaking_clusters;
    tbb::enumerable_thread_specific<ScalableVector<ClusterID>> tie_breaking_favored_clusters;
    CacheAlignedVector<LocalClusterSelectionState<ClusterID, EdgeWeight>> local_states;

    void free() {
      tie_breaking_clusters.clear();
      tie_breaking_favored_clusters.clear();
      local_states.clear();
    }
  };

  struct TwoPhaseStorage {
    ConcurrentRatingMap concurrent_rating_map;
    tbb::concurrent_vector<NodeID> nodes;

    void free() {
      concurrent_rating_map.free();
      nodes.clear();
      nodes.shrink_to_fit();
    }
  };

  struct PostprocessingStorage {
    StaticArray<std::uint8_t> moved;
    StaticArray<ClusterID> favored_clusters;

    void free() {
      moved.free();
      favored_clusters.free();
    }
  };

  RatingBuffers rating;
  ActiveSetStorage active_set;
  SelectionBuffers selection;
  TwoPhaseStorage two_phase;
  PostprocessingStorage postprocessing;

  void allocate(
      const NodeID num_nodes,
      const NodeID num_active_nodes,
      const ClusterID num_clusters,
      const ClusterID,
      const PassConfig<NodeID, ClusterID> &config
  ) {
    if (config.active_set.strategy == ActiveSetStrategy::LOCAL) {
      if (active_set.flags.size() < num_nodes) {
        active_set.flags.resize(num_nodes);
      }
    } else if (config.active_set.strategy == ActiveSetStrategy::GLOBAL) {
      if (active_set.flags.size() < num_active_nodes) {
        active_set.flags.resize(num_active_nodes);
      }
    }

    if (config.selection.track_favored_clusters) {
      if (postprocessing.favored_clusters.size() < num_active_nodes) {
        postprocessing.favored_clusters.resize(num_active_nodes);
      }
    }

    if (rating.maps.empty() || rating.map_capacity < num_clusters) {
      rating.maps = tbb::enumerable_thread_specific<RatingMap>([num_clusters] {
        return RatingMap(num_clusters);
      });
      rating.map_capacity = num_clusters;
    } else {
      for (auto &rating_map : rating.maps) {
        rating_map.change_max_size(num_clusters);
      }
    }

    if (selection.local_states.size() <
        static_cast<std::size_t>(tbb::this_task_arena::max_concurrency())) {
      selection.local_states.resize(tbb::this_task_arena::max_concurrency());
    }
  }

  void free() {
    rating.free();
    active_set.free();
    selection.free();
    two_phase.free();
    postprocessing.free();
  }
};

} // namespace kaminpar::lp
