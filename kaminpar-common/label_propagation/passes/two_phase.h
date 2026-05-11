/*******************************************************************************
 * Two-phase label propagation pass.
 *
 * @file:   two_phase.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <optional>
#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/label_propagation/kernel.h"
#include "kaminpar-common/label_propagation/postprocessing.h"

namespace kaminpar::lp {

template <typename Kernel, TieBreakingStrategy TieBreaking> class TwoPhasePass {
public:
  using NodeID = typename Kernel::NodeID;
  using NodeWeight = typename Kernel::NodeWeight;
  using EdgeWeight = typename Kernel::EdgeWeight;
  using ClusterID = typename Kernel::ClusterID;
  using ClusterWeight = typename Kernel::ClusterWeight;
  using Move = typename Kernel::Move;
  using Result = typename Kernel::Result;
  using Stats = typename Kernel::Stats;
  using RatingMap = typename Kernel::RatingMap;
  using ConcurrentRatingMap = typename Kernel::ConcurrentRatingMap;
  using SelectionContext = typename Kernel::SelectionContext;
  using TieBreakingBuffer = ScalableVector<ClusterID>;

  explicit TwoPhasePass(Kernel &kernel, const ExecutionConfig execution)
      : _kernel(kernel),
        _execution(execution) {
    _kernel.workspace().two_phase.nodes.clear();
  }

  class Local {
  public:
    Local(Kernel &kernel, const ExecutionConfig &execution, Stats &stats)
        : _kernel(kernel),
          _execution(execution),
          _stats(stats),
          _rand(Random::instance()),
          _tie_breaking_clusters(kernel.workspace().selection.tie_breaking_clusters.local()),
          _tie_breaking_favored_clusters(
              kernel.workspace().selection.tie_breaking_favored_clusters.local()
          ) {}

    [[nodiscard]] KAMINPAR_INLINE bool should_consider(const NodeID u) const {
      return _kernel.should_consider(u);
    }

    [[nodiscard]] KAMINPAR_INLINE std::optional<Move> find_best_move(const NodeID u) {
      if constexpr (Kernel::WorkspaceType::kSupportsTwoPhase) {
        const NodeWeight u_weight = _kernel.graph().node_weight(u);
        const ClusterID u_cluster = _kernel.labels().cluster(u);
        auto &map = rating_map();
        const std::size_t upper_bound_size = std::min<std::size_t>(
            {static_cast<std::size_t>(_kernel.graph().degree(u)),
             static_cast<std::size_t>(_kernel.initial_num_clusters()),
             _execution.large_map_threshold}
        );

        return map.execute(upper_bound_size, [&](auto &actual_map) -> std::optional<Move> {
          bool is_interface_node = false;
          const bool is_second_phase_node = _kernel.rate_neighbors_until(
              u, actual_map, _execution.large_map_threshold, is_interface_node
          );

          if (is_second_phase_node) [[unlikely]] {
            actual_map.clear();
            _kernel.workspace().two_phase.nodes.push_back(u);
            return std::nullopt;
          }

          _kernel.clear_active(u, is_interface_node);
          return _kernel.template select_move<TieBreaking>(
              u,
              u_weight,
              u_cluster,
              _rand,
              actual_map,
              _tie_breaking_clusters,
              _tie_breaking_favored_clusters
          );
        });
      } else {
        KASSERT(false, "two-phase label propagation is not supported by this workspace");
        __builtin_unreachable();
      }
    }

    KAMINPAR_INLINE std::pair<bool, bool> try_commit_move(const Move &move) {
      return _kernel.commit(move, _stats);
    }

    KAMINPAR_INLINE void handle_next_node(const NodeID u) {
      if (!should_consider(u)) {
        return;
      }

      ++_stats.processed_nodes;
      const auto move = find_best_move(u);
      if (move.has_value()) {
        try_commit_move(*move);
      }
    }

  private:
    KAMINPAR_INLINE RatingMap &rating_map() {
      if (_rating_map == nullptr) {
        _rating_map = &_kernel.workspace().rating.maps.local();
      }
      return *_rating_map;
    }

    Kernel &_kernel;
    const ExecutionConfig &_execution;
    Stats &_stats;
    Random &_rand;
    TieBreakingBuffer &_tie_breaking_clusters;
    TieBreakingBuffer &_tie_breaking_favored_clusters;
    RatingMap *_rating_map = nullptr;
  };

  [[nodiscard]] Local local() {
    return Local(_kernel, _execution, _stats.local());
  }

  KAMINPAR_INLINE void handle_next_node(const NodeID u) {
    auto local_pass = local();
    local_pass.handle_next_node(u);
  }

  [[nodiscard]] Result finish() {
    finish_second_phase(_stats.local());
    return _kernel.finish_pass(_stats);
  }

private:
  void finish_second_phase(Stats &stats) {
    if constexpr (Kernel::WorkspaceType::kSupportsTwoPhase) {
      auto &workspace = _kernel.workspace();
      const std::size_t num_clusters = _kernel.initial_num_clusters();
      if (workspace.two_phase.concurrent_rating_map.capacity() < num_clusters) {
        workspace.two_phase.concurrent_rating_map.resize(num_clusters);
      }

      if (!workspace.two_phase.nodes.empty() && _execution.relabel_before_second_phase) {
        relabel_clusters(_kernel);
      }

      auto &rand = Random::instance();
      for (const NodeID u : workspace.two_phase.nodes) {
        if (_kernel.neighbors().skip(u)) {
          continue;
        }

        const NodeWeight u_weight = _kernel.graph().node_weight(u);
        const ClusterID u_cluster = _kernel.labels().cluster(u);
        const Move move = find_best_move_second_phase(
            u, u_weight, u_cluster, rand, workspace.two_phase.concurrent_rating_map
        );
        const auto [moved, emptied] = _kernel.commit(move, stats);

        if (moved && _kernel.relabeled() && u < workspace.postprocessing.moved.size()) {
          workspace.postprocessing.moved[u] = 1;
        }
        (void)emptied;
      }

      workspace.two_phase.nodes.clear();
    } else {
      KASSERT(false, "two-phase label propagation is not supported by this workspace");
    }
  }

  [[nodiscard]] Move find_best_move_second_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ConcurrentRatingMap &map
  ) {
    auto &workspace = _kernel.workspace();
    const auto &config = _kernel.config();
    const ClusterWeight initial_cluster_weight = _kernel.weights().cluster_weight(u_cluster);

    const auto flush_local_rating_map = [&](auto &local_used_entries, auto &local_rating_map) {
      for (const auto [cluster, rating] : local_rating_map.entries()) {
        const EdgeWeight prev_rating = __atomic_fetch_add(&map[cluster], rating, __ATOMIC_RELAXED);

        if (prev_rating == 0) {
          local_used_entries.push_back(cluster);
        }
      }

      local_rating_map.clear();
    };

    bool is_interface_node = false;
    _kernel.graph().pfor_adjacent_nodes(
        u, config.nodes.max_neighbors, 2000, [&](auto &&pfor_adjacent_nodes) {
          auto &local_used_entries = map.local_used_entries();
          auto &local_rating_map = workspace.rating.maps.local().small_map();

          pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight w) {
            if (_kernel.neighbors().accept(u, v)) {
              const ClusterID v_cluster = _kernel.labels().cluster(v);
              local_rating_map[v_cluster] += w;

              if (local_rating_map.size() >= _execution.large_map_threshold) [[unlikely]] {
                flush_local_rating_map(local_used_entries, local_rating_map);
              }

              if (config.active_set.strategy == ActiveSetStrategy::LOCAL) {
                is_interface_node |= v >= _kernel.num_active_nodes();
              }
            }
          });
        }
    );

    tbb::parallel_for(workspace.rating.maps.range(), [&](auto &rating_maps) {
      auto &local_used_entries = map.local_used_entries();
      for (auto &rating_map : rating_maps) {
        auto &local_rating_map = rating_map.small_map();
        flush_local_rating_map(local_used_entries, local_rating_map);
      }
    });

    _kernel.clear_active(u, is_interface_node);

    const bool track_favored_cluster =
        config.selection.track_favored_clusters && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _kernel.weights().max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = config.selection.use_actual_gain ? map[u_cluster] : 0;

    map.iterate_and_reset([&](const auto i, auto &local_entries) {
      SelectionContext context{
          .rand = Random::instance(),
          .node = u,
          .node_weight = u_weight,
          .initial_cluster = u_cluster,
          .initial_cluster_weight = initial_cluster_weight,
          .gain_delta = gain_delta,
          .track_favored_cluster = track_favored_cluster,
      };

      const auto choice = _kernel.selector().template select<TieBreaking>(
          context,
          local_entries,
          workspace.selection.tie_breaking_clusters.local(),
          workspace.selection.tie_breaking_favored_clusters.local()
      );
      const EdgeWeight local_favored_cluster_gain = map[choice.favored_cluster];

      workspace.selection.local_states[i] = {
          choice.best_gain,
          choice.best_cluster,
          local_favored_cluster_gain,
          choice.favored_cluster,
      };
    });

    ClusterID favored_cluster = u_cluster;
    ClusterID best_cluster = u_cluster;
    EdgeWeight best_gain = 0;

    if constexpr (TieBreaking == TieBreakingStrategy::UNIFORM) {
      auto &tie_breaking_clusters = workspace.selection.tie_breaking_clusters.local();
      auto &tie_breaking_favored_clusters =
          workspace.selection.tie_breaking_favored_clusters.local();

      EdgeWeight favored_cluster_gain = 0;
      for (auto &local_state : workspace.selection.local_states) {
        if (local_state.best_gain > best_gain) {
          best_gain = local_state.best_gain;
          best_cluster = local_state.best_cluster;

          tie_breaking_clusters.clear();
          tie_breaking_clusters.push_back(local_state.best_cluster);
        } else if (local_state.best_gain == best_gain) {
          tie_breaking_clusters.push_back(local_state.best_cluster);
        }

        if (track_favored_cluster) {
          if (local_state.favored_cluster_gain > favored_cluster_gain) {
            favored_cluster_gain = local_state.favored_cluster_gain;
            favored_cluster = local_state.favored_cluster;

            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(local_state.favored_cluster);
          } else if (local_state.favored_cluster_gain == favored_cluster_gain) {
            tie_breaking_favored_clusters.push_back(local_state.favored_cluster);
          }
        }

        local_state.best_gain = -1;
        local_state.favored_cluster_gain = -1;
      }

      if (tie_breaking_clusters.size() > 1) {
        const ClusterID i = rand.random_index(0, tie_breaking_clusters.size());
        best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const ClusterID i = rand.random_index(0, tie_breaking_favored_clusters.size());
        favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();
    } else {
      EdgeWeight favored_cluster_gain = 0;
      for (auto &local_state : workspace.selection.local_states) {
        if (local_state.best_gain > best_gain) {
          best_gain = local_state.best_gain;
          best_cluster = local_state.best_cluster;
        }

        if (track_favored_cluster && local_state.favored_cluster_gain > favored_cluster_gain) {
          favored_cluster_gain = local_state.favored_cluster_gain;
          favored_cluster = local_state.favored_cluster;
        }

        local_state.best_gain = -1;
        local_state.favored_cluster_gain = -1;
      }
    }

    if (track_favored_cluster && best_cluster == u_cluster) {
      workspace.postprocessing.favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = best_gain - map[u_cluster];
    return {
        .node = u,
        .node_weight = u_weight,
        .old_cluster = u_cluster,
        .new_cluster = best_cluster,
        .gain = actual_gain,
        .valid = true,
    };
  }

  Kernel &_kernel;
  ExecutionConfig _execution;
  tbb::enumerable_thread_specific<Stats> _stats;
};

} // namespace kaminpar::lp
