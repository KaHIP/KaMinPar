/*******************************************************************************
 * Growing-hash-table label propagation pass.
 *
 * @file:   growing_hash_tables.h
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/label_propagation/kernel.h"

namespace kaminpar::lp {

template <typename Kernel, ActiveSetStrategy ActiveSet, TieBreakingStrategy TieBreaking>
class GrowingHashTablePass {
public:
  using NodeID = typename Kernel::NodeID;
  using ClusterID = typename Kernel::ClusterID;
  using Move = typename Kernel::Move;
  using Result = typename Kernel::Result;
  using Stats = typename Kernel::Stats;
  using GrowingRatingMap = typename Kernel::GrowingRatingMap;
  using TieBreakingBuffer = ScalableVector<ClusterID>;

  explicit GrowingHashTablePass(Kernel &kernel) : _kernel(kernel) {}

  class Local {
  public:
    Local(Kernel &kernel, Stats &stats)
        : _kernel(kernel),
          _stats(stats),
          _rand(Random::instance()),
          _rating_map(kernel.workspace().rating.growing_maps.local()),
          _tie_breaking_clusters(kernel.workspace().selection.tie_breaking_clusters.local()),
          _tie_breaking_favored_clusters(
              kernel.workspace().selection.tie_breaking_favored_clusters.local()
          ) {}

    [[nodiscard]] KAMINPAR_INLINE bool should_consider(const NodeID u) const {
      return _kernel.template should_consider<ActiveSet>(u);
    }

    [[nodiscard]] KAMINPAR_INLINE Move find_best_move(const NodeID u) {
      return _kernel.template find_best_move<ActiveSet, TieBreaking>(
          u, _rand, _rating_map, _tie_breaking_clusters, _tie_breaking_favored_clusters
      );
    }

    KAMINPAR_INLINE std::pair<bool, bool> try_commit_move(const Move &move) {
      return _kernel.template commit<ActiveSet>(move, _stats);
    }

    KAMINPAR_INLINE void handle_next_node(const NodeID u) {
      if (!should_consider(u)) {
        return;
      }

      ++_stats.processed_nodes;
      const auto u_weight = _kernel.graph().node_weight(u);
      const auto u_cluster = _kernel.labels().cluster(u);
      const auto [best_cluster, gain] = _kernel.template find_best_target<ActiveSet, TieBreaking>(
          u,
          u_weight,
          u_cluster,
          _rand,
          _rating_map,
          _tie_breaking_clusters,
          _tie_breaking_favored_clusters
      );
      _kernel.template commit<ActiveSet>(u, u_weight, u_cluster, best_cluster, gain, _stats);
    }

  private:
    Kernel &_kernel;
    Stats &_stats;
    Random &_rand;
    GrowingRatingMap &_rating_map;
    TieBreakingBuffer &_tie_breaking_clusters;
    TieBreakingBuffer &_tie_breaking_favored_clusters;
  };

  [[nodiscard]] Local local() {
    return Local(_kernel, _stats.local());
  }

  class BufferedLocal {
  public:
    BufferedLocal(Kernel &kernel, Stats &target_stats)
        : _target_stats(target_stats),
          _local(kernel, _stats) {}

    BufferedLocal(const BufferedLocal &) = delete;
    BufferedLocal &operator=(const BufferedLocal &) = delete;
    BufferedLocal(BufferedLocal &&) = delete;
    BufferedLocal &operator=(BufferedLocal &&) = delete;

    ~BufferedLocal() {
      _target_stats += _stats;
    }

    KAMINPAR_INLINE void handle_next_node(const NodeID u) {
      _local.handle_next_node(u);
    }

  private:
    Stats &_target_stats;
    Stats _stats;
    Local _local;
  };

  [[nodiscard]] BufferedLocal buffered_local() {
    return BufferedLocal(_kernel, _stats.local());
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
