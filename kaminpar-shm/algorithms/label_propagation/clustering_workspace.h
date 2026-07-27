/*******************************************************************************
 * Reusable storage for parallel label-propagation clustering.
 *
 * @file:   clustering_workspace.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstddef>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/algorithms/label_propagation/parallel_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/algorithms/label_propagation/round_statistics.h"

#include "kaminpar-common/datastructures/cache_aligned_vector.h"

namespace kaminpar::shm::lp {

using ClusteringRatingMaps = RatingMapPool<EdgeWeight, NodeID>;

class ClusteringWorkspace {
public:
  void ensure_capacity(const NodeID num_nodes) {
    rating_maps.ensure_capacity(num_nodes);
    const std::size_t concurrency = tbb::this_task_arena::max_concurrency();
    if (local_selections.size() < concurrency) {
      local_selections.resize(
          concurrency,
          ClusteringSelector::RatedSelection{
              .cluster = 0,
              .rating = -1,
              .favored_cluster = 0,
              .favored_rating = -1,
          }
      );
    }
  }

  void begin_round() {
    statistics.clear();
  }

  void free() {
    state.free();
    rating_maps.free();
    parallel_ratings.free();
    deferred_nodes.clear();
    deferred_nodes.shrink_to_fit();
    local_selections.clear();
    local_selections.shrink_to_fit();
    statistics.clear();
  }

  ClusteringState state;
  ClusteringRatingMaps rating_maps;
  ParallelRatingMap<EdgeWeight, NodeID> parallel_ratings;
  tbb::concurrent_vector<NodeID> deferred_nodes;
  CacheAlignedVector<ClusteringSelector::RatedSelection> local_selections;
  RoundStatistics statistics;
};

} // namespace kaminpar::shm::lp
