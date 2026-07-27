/*******************************************************************************
 * Reusable storage for balanced label-propagation refinement.
 *
 * @file:   balanced_workspace.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/algorithms/label_propagation/balanced_state.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/algorithms/label_propagation/round_statistics.h"

#include "kaminpar-common/datastructures/sparse_map.h"

namespace kaminpar::shm::lp {

using BalancedRatingMaps = RatingMapPool<EdgeWeight, BlockID, rm_backyard::SparseMap>;

class BalancedWorkspace {
public:
  void ensure_capacity(const BlockID num_blocks) {
    rating_maps.ensure_capacity(num_blocks);
  }

  void begin_round() {
    statistics.clear();
  }

  void free() {
    state.free();
    rating_maps.free();
    statistics.clear();
  }

  BalancedState state;
  BalancedRatingMaps rating_maps;
  RoundStatistics statistics;
};

} // namespace kaminpar::shm::lp
