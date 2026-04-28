/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   core.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/label_propagation/active_set.h"
#include "kaminpar-common/label_propagation/cluster_chooser.h"
#include "kaminpar-common/label_propagation/move.h"
#include "kaminpar-common/label_propagation/rating_accumulator.h"
#include "kaminpar-common/label_propagation/stores.h"
#include "kaminpar-common/label_propagation/types.h"
#include "kaminpar-common/label_propagation/workspace.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::lp {

template <
    typename Graph,
    typename LabelStore,
    typename WeightStore,
    typename ClusterSelector,
    typename NeighborPolicy,
    typename Workspace>
class LabelPropagationCore {
  SET_STATISTICS_FROM_GLOBAL();

public:
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using ClusterID = typename LabelStore::ClusterIDType;
  using ClusterWeight = typename WeightStore::ClusterWeightType;
  using RatingMap = typename Workspace::RatingMapType;
  using GrowingRatingMap = typename Workspace::GrowingRatingMapType;
  using ConcurrentRatingMap = typename Workspace::ConcurrentRatingMapType;
  using SelectionContext = NodeContext<NodeID, NodeWeight, ClusterID, ClusterWeight, EdgeWeight>;
  using Result = PassResult<NodeID, ClusterID, EdgeWeight>;
  using ActiveSet = ActiveSetView<NodeID, Graph, NeighborPolicy, Workspace>;
  using Move = NodeMove<NodeID, NodeWeight, ClusterID, EdgeWeight>;
  using Stats = PassStats<NodeID, ClusterID, EdgeWeight>;
  using MoveApplier = lp::
      MoveApplier<NodeID, NodeWeight, ClusterID, EdgeWeight, LabelStore, WeightStore, ActiveSet>;
  using RatingAccumulator = lp::RatingAccumulator<NodeID, Graph, LabelStore, NeighborPolicy>;

  LabelPropagationCore(
      const Graph &graph,
      LabelStore &labels,
      WeightStore &weights,
      ClusterSelector &selector,
      NeighborPolicy &neighbors,
      Workspace &workspace,
      PassConfig<NodeID, ClusterID> config
  )
      : _graph(graph),
        _labels(labels),
        _weights(weights),
        _selector(selector),
        _neighbors(neighbors),
        _workspace(workspace),
        _config(config),
        _active_set(graph, neighbors, workspace, _config.active_set),
        _move_applier(labels, weights, _active_set, _config.stopping),
        _rating_accumulator(graph, labels, neighbors, _config.nodes, _config.active_set) {}

  void set_config(const PassConfig<NodeID, ClusterID> config) {
    _config = config;
  }

  [[nodiscard]] const PassConfig<NodeID, ClusterID> &config() const {
    return _config;
  }

  void initialize(const Initialization<NodeID, ClusterID> init) {
    _num_nodes = init.num_nodes;
    _num_active_nodes = init.num_active_nodes;
    _prev_num_clusters = _num_clusters;
    _num_clusters = init.num_clusters;
    _initial_num_clusters = init.num_clusters;
    _current_num_clusters = init.num_clusters;
    _relabeled = false;
    _workspace.allocate(_num_nodes, _num_active_nodes, _num_clusters, _prev_num_clusters, _config);
    reset_state();
  }

  void clear_iteration_order_cache() {}

  [[nodiscard]] ClusterID current_num_clusters() const {
    return _current_num_clusters;
  }

  [[nodiscard]] bool should_stop() const {
    return _config.stopping.track_cluster_count &&
           _current_num_clusters <= _config.stopping.desired_clusters;
  }

  class Pass {
  public:
    explicit Pass(LabelPropagationCore &core) : _core(core) {}

    class Local {
    public:
      Local(LabelPropagationCore &core, Stats &stats)
          : _core(core),
            _stats(stats),
            _rand(Random::instance()),
            _tie_breaking_clusters(core._workspace.tie_breaking_clusters_ets.local()),
            _tie_breaking_favored_clusters(
                core._workspace.tie_breaking_favored_clusters_ets.local()
            ) {}

      [[nodiscard]] KAMINPAR_LP_INLINE bool should_consider(const NodeID u) const {
        return _core.should_consider(u);
      }

      [[nodiscard]] KAMINPAR_LP_INLINE Move find_best_move(const NodeID u) {
        return _core.find_best_move(
            u, _rand, rating_map(), _tie_breaking_clusters, _tie_breaking_favored_clusters
        );
      }

      KAMINPAR_LP_INLINE std::pair<bool, bool> try_commit_move(const Move &move) {
        return _core.try_commit_move(move, _stats);
      }

      KAMINPAR_LP_INLINE void handle_next_node(const NodeID u) {
        switch (_core._config.rating.strategy) {
        case RatingMapStrategy::GROWING_HASH_TABLES:
          handle_next_node_with_rating_strategy<RatingMapStrategy::GROWING_HASH_TABLES>(u);
          break;

        case RatingMapStrategy::SINGLE_PHASE:
          handle_next_node_with_rating_strategy<RatingMapStrategy::SINGLE_PHASE>(u);
          break;

        case RatingMapStrategy::TWO_PHASE:
          handle_next_node_with_rating_strategy<RatingMapStrategy::TWO_PHASE>(u);
          break;
        }
      }

      template <RatingMapStrategy Strategy>
      KAMINPAR_LP_INLINE void handle_next_node_with_rating_strategy(const NodeID u) {
        switch (_core._config.selection.tie_breaking_strategy) {
        case TieBreakingStrategy::GEOMETRIC:
          handle_next_node<Strategy, TieBreakingStrategy::GEOMETRIC>(u);
          break;

        case TieBreakingStrategy::UNIFORM:
          handle_next_node<Strategy, TieBreakingStrategy::UNIFORM>(u);
          break;
        }
      }

      template <RatingMapStrategy Strategy, TieBreakingStrategy TieBreaking>
      KAMINPAR_LP_INLINE void handle_next_node(const NodeID u) {
        _current_node = u;
        if (!should_consider(u)) {
          return;
        }

        ++_stats.processed_nodes;

        if constexpr (Strategy == RatingMapStrategy::GROWING_HASH_TABLES) {
          handle_node_with_map<TieBreaking>(growing_rating_map());
        } else if constexpr (Strategy == RatingMapStrategy::SINGLE_PHASE) {
          handle_node_with_map<TieBreaking>(rating_map());
        } else if constexpr (Strategy == RatingMapStrategy::TWO_PHASE) {
          handle_first_phase_node<TieBreaking>();
        }
      }

    private:
      template <TieBreakingStrategy TieBreaking, typename LocalRatingMap>
      KAMINPAR_LP_INLINE void handle_node_with_map(LocalRatingMap &rating_map) {
        const Move move = _core.template find_best_move<TieBreaking>(
            _current_node, _rand, rating_map, _tie_breaking_clusters, _tie_breaking_favored_clusters
        );
        _core.try_commit_move(move, _stats);
      }

      template <TieBreakingStrategy TieBreaking> KAMINPAR_LP_INLINE void handle_first_phase_node() {
        if constexpr (Workspace::kSupportsTwoPhase) {
          const NodeID u = _current_node;
          const NodeWeight u_weight = _core._graph.node_weight(u);
          const ClusterID u_cluster = _core._labels.cluster(u);
          auto &map = rating_map();
          const std::size_t upper_bound_size = std::min<std::size_t>(
              {_core._graph.degree(u),
               _core._initial_num_clusters,
               _core._config.rating.large_map_threshold}
          );

          const auto maybe_move = map.execute(upper_bound_size, [&](auto &actual_map) {
            return _core.template find_best_move_first_phase<TieBreaking>(
                u,
                u_weight,
                u_cluster,
                _rand,
                actual_map,
                _tie_breaking_clusters,
                _tie_breaking_favored_clusters
            );
          });

          if (maybe_move.has_value()) {
            _core.try_commit_move(*maybe_move, _stats);
          }
        } else {
          KASSERT(false, "two-phase label propagation is not supported by this workspace");
        }
      }

      KAMINPAR_LP_INLINE RatingMap &rating_map() {
        if (_rating_map == nullptr) {
          _rating_map = &_core._workspace.rating_map_ets.local();
        }
        return *_rating_map;
      }

      KAMINPAR_LP_INLINE GrowingRatingMap &growing_rating_map() {
        if (_growing_rating_map == nullptr) {
          _growing_rating_map = &_core._workspace.growing_rating_map_ets.local();
        }
        return *_growing_rating_map;
      }

      friend class Pass;

      LabelPropagationCore &_core;
      Stats &_stats;
      Random &_rand;
      ScalableVector<ClusterID> &_tie_breaking_clusters;
      ScalableVector<ClusterID> &_tie_breaking_favored_clusters;
      RatingMap *_rating_map = nullptr;
      GrowingRatingMap *_growing_rating_map = nullptr;
      NodeID _current_node = 0;
    };

    [[nodiscard]] Local local() {
      return Local(_core, _stats.local());
    }

    void handle_next_node(const NodeID u) {
      auto local_pass = local();
      local_pass.handle_next_node(u);
    }

    [[nodiscard]] Result finish() {
      if (_core._config.rating.strategy == RatingMapStrategy::TWO_PHASE) {
        _core.finish_second_phase(_stats.local());
      }

      Result result;
      for (const Stats &local_stats : _stats) {
        result.processed_nodes += local_stats.processed_nodes;
        result.moved_nodes += local_stats.moved_nodes;
        result.removed_clusters += local_stats.removed_clusters;
        result.expected_total_gain += local_stats.expected_total_gain;
      }

      _core._current_num_clusters -= result.removed_clusters;
      _core._expected_total_gain += result.expected_total_gain;
      return result;
    }

  private:
    LabelPropagationCore &_core;
    tbb::enumerable_thread_specific<Stats> _stats;
  };

  [[nodiscard]] Pass begin_pass() {
    _workspace.second_phase_nodes.clear();
    return Pass(*this);
  }

  template <typename LocalRatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE Move find_best_move(
      const NodeID u,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    switch (_config.selection.tie_breaking_strategy) {
    case TieBreakingStrategy::GEOMETRIC:
      return find_best_move<TieBreakingStrategy::GEOMETRIC>(
          u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    case TieBreakingStrategy::UNIFORM:
      return find_best_move<TieBreakingStrategy::UNIFORM>(
          u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    }
  }

  template <TieBreakingStrategy TieBreaking, typename LocalRatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE Move find_best_move(
      const NodeID u,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const NodeWeight u_weight = _graph.node_weight(u);
    const ClusterID u_cluster = _labels.cluster(u);
    return find_best_move_impl<TieBreaking>(
        u,
        u_weight,
        u_cluster,
        rand,
        rating_map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );
  }

  [[nodiscard]] KAMINPAR_LP_INLINE Move find_best_move(const NodeID u, Random &rand) {
    auto &rating_map = _workspace.rating_map_ets.local();
    auto &tie_breaking_clusters = _workspace.tie_breaking_clusters_ets.local();
    auto &tie_breaking_favored_clusters = _workspace.tie_breaking_favored_clusters_ets.local();
    return find_best_move(
        u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  KAMINPAR_LP_INLINE std::pair<bool, bool> try_commit_move(const Move &move, Stats &stats) {
    return _move_applier.try_commit(move, stats);
  }

  KAMINPAR_LP_INLINE void activate_neighbors_of_ghost_node(const NodeID u) {
    _active_set.activate_neighbors_of_ghost_node(u);
  }

#include "kaminpar-common/label_propagation/postprocessing_public.inc"

private:
  [[nodiscard]] KAMINPAR_LP_INLINE bool should_consider(const NodeID u) const {
    if (u >= _num_active_nodes) {
      return false;
    }
    if (_neighbors.skip(u)) {
      return false;
    }
    if (_graph.degree(u) >= _config.nodes.max_degree) {
      return false;
    }
    if (!_active_set.is_active(u)) {
      return false;
    }
    return true;
  }

  template <TieBreakingStrategy TieBreaking, typename LocalRatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE Move find_best_move_impl(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    auto action = [&](auto &map) {
      return compute_best_move_from_map<TieBreaking>(
          u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    };

    if constexpr (std::is_same_v<LocalRatingMap, GrowingRatingMap>) {
      return action(rating_map);
    } else {
      const std::size_t upper_bound_size =
          std::min<ClusterID>(_graph.degree(u), _initial_num_clusters);
      return rating_map.execute(upper_bound_size, action);
    }
  }

  template <TieBreakingStrategy TieBreaking, typename LocalRatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE std::optional<Move> find_best_move_first_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      LocalRatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    bool is_interface_node = false;
    const bool is_second_phase_node = _rating_accumulator.rate_neighbors_until(
        u, map, _num_active_nodes, _config.rating.large_map_threshold, is_interface_node
    );

    if (is_second_phase_node) [[unlikely]] {
      map.clear();
      _workspace.second_phase_nodes.push_back(u);
      return std::nullopt;
    }

    clear_active(u, is_interface_node);

    return compute_best_move_after_rating<TieBreaking>(
        u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  template <TieBreakingStrategy TieBreaking, typename ActualMap>
  [[nodiscard]] KAMINPAR_LP_INLINE Move compute_best_move_from_map(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    bool is_interface_node = false;
    _rating_accumulator.rate_neighbors(u, map, _num_active_nodes, is_interface_node);

    clear_active(u, is_interface_node);

    return compute_best_move_after_rating<TieBreaking>(
        u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  template <TieBreakingStrategy TieBreaking, typename ActualMap>
  [[nodiscard]] KAMINPAR_LP_INLINE Move compute_best_move_after_rating(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = _weights.cluster_weight(u_cluster);
    const bool track_favored_cluster =
        _config.selection.track_favored_clusters && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _config.selection.use_actual_gain ? map[u_cluster] : 0;
    SelectionContext context{
        .rand = rand,
        .node = u,
        .node_weight = u_weight,
        .initial_cluster = u_cluster,
        .initial_cluster_weight = initial_cluster_weight,
        .gain_delta = gain_delta,
        .track_favored_cluster = track_favored_cluster,
    };

    const auto choice = _selector.template select<TieBreaking>(
        context, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );

    if (track_favored_cluster && choice.best_cluster == context.initial_cluster) {
      _workspace.favored_clusters[u] = choice.favored_cluster;
    }

    const EdgeWeight actual_gain = choice.best_gain - map[context.initial_cluster];
    map.clear();
    return {
        .node = u,
        .node_weight = u_weight,
        .old_cluster = u_cluster,
        .new_cluster = choice.best_cluster,
        .gain = actual_gain,
        .valid = true,
    };
  }

  KAMINPAR_LP_INLINE void clear_active(const NodeID u, const bool is_interface_node) {
    _active_set.clear(u, is_interface_node);
  }

  void finish_second_phase(Stats &stats) {
    switch (_config.selection.tie_breaking_strategy) {
    case TieBreakingStrategy::GEOMETRIC:
      return finish_second_phase<TieBreakingStrategy::GEOMETRIC>(stats);
    case TieBreakingStrategy::UNIFORM:
      return finish_second_phase<TieBreakingStrategy::UNIFORM>(stats);
    }
  }

  template <TieBreakingStrategy TieBreaking> void finish_second_phase(Stats &stats) {
    if constexpr (Workspace::kSupportsTwoPhase) {
      const std::size_t num_clusters = _initial_num_clusters;
      if (_workspace.concurrent_rating_map.capacity() < num_clusters) {
        _workspace.concurrent_rating_map.resize(num_clusters);
      }

      if (!_workspace.second_phase_nodes.empty() && _config.rating.relabel_before_second_phase) {
        relabel_clusters();
      }

      auto &rand = Random::instance();
      for (const NodeID u : _workspace.second_phase_nodes) {
        if (_neighbors.skip(u)) {
          continue;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        const ClusterID u_cluster = _labels.cluster(u);
        const Move move = find_best_move_second_phase<TieBreaking>(
            u, u_weight, u_cluster, rand, _workspace.concurrent_rating_map
        );
        const auto [moved, emptied] = try_commit_move(move, stats);

        if (moved && _relabeled && u < _workspace.moved.size()) {
          _workspace.moved[u] = 1;
        }
        (void)emptied;
      }

      _workspace.second_phase_nodes.clear();
    } else {
      KASSERT(false, "two-phase label propagation is not supported by this workspace");
    }
  }

  template <TieBreakingStrategy TieBreaking>
  [[nodiscard]] Move find_best_move_second_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ConcurrentRatingMap &map
  ) {
    const ClusterWeight initial_cluster_weight = _weights.cluster_weight(u_cluster);

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
    _graph.pfor_adjacent_nodes(
        u, _config.nodes.max_neighbors, 2000, [&](auto &&pfor_adjacent_nodes) {
          auto &local_used_entries = map.local_used_entries();
          auto &local_rating_map = _workspace.rating_map_ets.local().small_map();

          pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight w) {
            if (_neighbors.accept(u, v)) {
              const ClusterID v_cluster = _labels.cluster(v);
              local_rating_map[v_cluster] += w;

              if (local_rating_map.size() >= _config.rating.large_map_threshold) [[unlikely]] {
                flush_local_rating_map(local_used_entries, local_rating_map);
              }

              if (_config.active_set.strategy == ActiveSetStrategy::LOCAL) {
                is_interface_node |= v >= _num_active_nodes;
              }
            }
          });
        }
    );

    tbb::parallel_for(_workspace.rating_map_ets.range(), [&](auto &rating_maps) {
      auto &local_used_entries = map.local_used_entries();
      for (auto &rating_map : rating_maps) {
        auto &local_rating_map = rating_map.small_map();
        flush_local_rating_map(local_used_entries, local_rating_map);
      }
    });

    clear_active(u, is_interface_node);

    const bool track_favored_cluster =
        _config.selection.track_favored_clusters && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _config.selection.use_actual_gain ? map[u_cluster] : 0;

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

      const auto choice = _selector.template select<TieBreaking>(
          context,
          local_entries,
          _workspace.tie_breaking_clusters_ets.local(),
          _workspace.tie_breaking_favored_clusters_ets.local()
      );
      const EdgeWeight local_favored_cluster_gain = map[choice.favored_cluster];

      _workspace.local_cluster_selection_states[i] = {
          choice.best_gain,
          choice.best_cluster,
          local_favored_cluster_gain,
          choice.favored_cluster,
      };
    });

    ClusterID favored_cluster = u_cluster;
    ClusterID best_cluster = u_cluster;
    EdgeWeight best_gain = 0;

    if (_config.selection.tie_breaking_strategy == TieBreakingStrategy::UNIFORM) {
      auto &tie_breaking_clusters = _workspace.tie_breaking_clusters_ets.local();
      auto &tie_breaking_favored_clusters = _workspace.tie_breaking_favored_clusters_ets.local();

      EdgeWeight favored_cluster_gain = 0;
      for (auto &local_state : _workspace.local_cluster_selection_states) {
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
      for (auto &local_state : _workspace.local_cluster_selection_states) {
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
      _workspace.favored_clusters[u] = favored_cluster;
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

#include "kaminpar-common/label_propagation/postprocessing_private.inc"

  const Graph &_graph;
  LabelStore &_labels;
  WeightStore &_weights;
  ClusterSelector &_selector;
  NeighborPolicy &_neighbors;
  Workspace &_workspace;
  PassConfig<NodeID, ClusterID> _config;
  ActiveSet _active_set;
  MoveApplier _move_applier;
  RatingAccumulator _rating_accumulator;

  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
  ClusterID _prev_num_clusters = 0;
  ClusterID _initial_num_clusters = 0;
  parallel::Atomic<ClusterID> _current_num_clusters = 0;
  parallel::Atomic<EdgeWeight> _expected_total_gain = 0;
  bool _relabeled = false;
};

template <
    RatingMapStrategy Strategy,
    TieBreakingStrategy TieBreaking,
    typename Order,
    typename Core,
    typename Pass>
void run_iteration_with_strategy(Order &order, Core &lp, Pass &pass) {
  auto make_local = [&] {
    return pass.local();
  };
  auto handle_node = [](auto &local, const auto node) {
    local.template handle_next_node<Strategy, TieBreaking>(node);
  };
  auto should_skip = [&] {
    return lp.should_stop();
  };

  if constexpr (requires {
                  order.parallel_for_each_with_local(make_local, handle_node, should_skip);
                }) {
    order.parallel_for_each_with_local(make_local, handle_node, should_skip);
  } else if constexpr (requires { order.parallel_for_each_with_local(make_local, handle_node); }) {
    order.parallel_for_each_with_local(make_local, handle_node);
  } else {
    order.parallel_for_each([&](const auto node) {
      if (lp.should_stop()) {
        return;
      }
      auto local = pass.local();
      local.template handle_next_node<Strategy, TieBreaking>(node);
    });
  }
}

template <RatingMapStrategy Strategy, typename Order, typename Core, typename Pass>
void run_iteration_with_rating_strategy(Order &order, Core &lp, Pass &pass) {
  switch (lp.config().selection.tie_breaking_strategy) {
  case TieBreakingStrategy::GEOMETRIC:
    run_iteration_with_strategy<Strategy, TieBreakingStrategy::GEOMETRIC>(order, lp, pass);
    break;

  case TieBreakingStrategy::UNIFORM:
    run_iteration_with_strategy<Strategy, TieBreakingStrategy::UNIFORM>(order, lp, pass);
    break;
  }
}

template <typename Order, typename Core>
typename Core::Result run_iteration(Order &order, Core &lp) {
  auto pass = lp.begin_pass();

  switch (lp.config().rating.strategy) {
  case RatingMapStrategy::GROWING_HASH_TABLES:
    run_iteration_with_rating_strategy<RatingMapStrategy::GROWING_HASH_TABLES>(order, lp, pass);
    break;

  case RatingMapStrategy::SINGLE_PHASE:
    run_iteration_with_rating_strategy<RatingMapStrategy::SINGLE_PHASE>(order, lp, pass);
    break;

  case RatingMapStrategy::TWO_PHASE:
    run_iteration_with_rating_strategy<RatingMapStrategy::TWO_PHASE>(order, lp, pass);
    break;
  }

  return pass.finish();
}

} // namespace kaminpar::lp
