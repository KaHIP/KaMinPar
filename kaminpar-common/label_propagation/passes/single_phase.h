/*******************************************************************************
 * Single-phase label propagation pass.
 *
 * @file:   single_phase.h
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/label_propagation/kernel.h"

namespace kaminpar::lp {

template <typename Kernel, TieBreakingStrategy TieBreaking> class SinglePhasePass {
public:
  using NodeID = typename Kernel::NodeID;
  using ClusterID = typename Kernel::ClusterID;
  using Move = typename Kernel::Move;
  using Result = typename Kernel::Result;
  using Stats = typename Kernel::Stats;
  using RatingMap = typename Kernel::RatingMap;
  using TieBreakingBuffer = ScalableVector<ClusterID>;

  explicit SinglePhasePass(Kernel &kernel) : _kernel(kernel) {}

  class Local {
  public:
    Local(Kernel &kernel, Stats &stats)
        : _kernel(kernel),
          _stats(stats),
          _rand(Random::instance()),
          _tie_breaking_clusters(kernel.workspace().selection.tie_breaking_clusters.local()),
          _tie_breaking_favored_clusters(
              kernel.workspace().selection.tie_breaking_favored_clusters.local()
          ) {}

    [[nodiscard]] KAMINPAR_INLINE bool should_consider(const NodeID u) const {
      return _kernel.should_consider(u);
    }

    [[nodiscard]] KAMINPAR_INLINE Move find_best_move(const NodeID u) {
      return _kernel.template find_best_move<TieBreaking>(
          u, _rand, rating_map(), _tie_breaking_clusters, _tie_breaking_favored_clusters
      );
    }

    KAMINPAR_INLINE std::pair<bool, bool> try_commit_move(const Move &move) {
      return _kernel.commit(move, _stats);
    }

    KAMINPAR_INLINE void handle_next_node(const NodeID u) {
      if (!should_consider(u)) {
        return;
      }

      ++_stats.processed_nodes;
      try_commit_move(find_best_move(u));
    }

  private:
    KAMINPAR_INLINE RatingMap &rating_map() {
      if (_rating_map == nullptr) {
        _rating_map = &_kernel.workspace().rating.maps.local();
      }
      return *_rating_map;
    }

    Kernel &_kernel;
    Stats &_stats;
    Random &_rand;
    TieBreakingBuffer &_tie_breaking_clusters;
    TieBreakingBuffer &_tie_breaking_favored_clusters;
    RatingMap *_rating_map = nullptr;
  };

  [[nodiscard]] Local local() {
    return Local(_kernel, _stats.local());
  }

  KAMINPAR_INLINE void handle_next_node(const NodeID u) {
    auto local_pass = local();
    local_pass.handle_next_node(u);
  }

  [[nodiscard]] Result finish() {
    return _kernel.finish_pass(_stats);
  }

private:
  Kernel &_kernel;
  tbb::enumerable_thread_specific<Stats> _stats;
};

} // namespace kaminpar::lp
