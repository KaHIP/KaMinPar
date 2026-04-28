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
  using SelectionState = ClusterSelectionState<NodeID, ClusterID, ClusterWeight, EdgeWeight>;
  using Result = PassResult<NodeID, ClusterID, EdgeWeight>;

  struct PassStats {
    NodeID processed_nodes = 0;
    NodeID moved_nodes = 0;
    ClusterID removed_clusters = 0;
    EdgeWeight expected_total_gain = 0;
  };

  struct BestMove {
    NodeID node;
    NodeWeight node_weight;
    ClusterID old_cluster;
    ClusterID new_cluster;
    EdgeWeight gain;
    bool valid = false;
  };

  LabelPropagationCore(
      const Graph &graph,
      LabelStore &labels,
      WeightStore &weights,
      ClusterSelector &selector,
      NeighborPolicy &neighbors,
      Workspace &workspace,
      Options<NodeID, ClusterID> options
  )
      : _graph(graph),
        _labels(labels),
        _weights(weights),
        _selector(selector),
        _neighbors(neighbors),
        _workspace(workspace),
        _options(options) {}

  void set_options(const Options<NodeID, ClusterID> options) {
    _options = options;
  }

  [[nodiscard]] const Options<NodeID, ClusterID> &options() const {
    return _options;
  }

  void initialize(const Initialization<NodeID, ClusterID> init) {
    _num_nodes = init.num_nodes;
    _num_active_nodes = init.num_active_nodes;
    _prev_num_clusters = _num_clusters;
    _num_clusters = init.num_clusters;
    _initial_num_clusters = init.num_clusters;
    _current_num_clusters = init.num_clusters;
    _relabeled = false;
    _workspace.allocate(_num_nodes, _num_active_nodes, _num_clusters, _prev_num_clusters, _options);
    reset_state();
  }

  void clear_iteration_order_cache() {}

  [[nodiscard]] ClusterID current_num_clusters() const {
    return _current_num_clusters;
  }

  [[nodiscard]] bool should_stop() const {
    return _options.track_cluster_count && _current_num_clusters <= _options.desired_num_clusters;
  }

  class Pass {
  public:
    explicit Pass(LabelPropagationCore &core) : _core(core) {}

    class Local {
    public:
      Local(LabelPropagationCore &core, PassStats &stats)
          : _core(core),
            _stats(stats),
            _rand(Random::instance()),
            _tie_breaking_clusters(core._workspace.tie_breaking_clusters_ets.local()),
            _tie_breaking_favored_clusters(
                core._workspace.tie_breaking_favored_clusters_ets.local()
            ) {}

      [[nodiscard]] bool should_consider(const NodeID u) const {
        return _core.should_consider(u);
      }

      [[nodiscard]] BestMove find_best_move(const NodeID u) {
        return _core.find_best_move(
            u, _rand, rating_map(), _tie_breaking_clusters, _tie_breaking_favored_clusters
        );
      }

      std::pair<bool, bool> try_commit_move(const BestMove &move) {
        return _core.try_commit_move(move, _stats);
      }

      void handle_next_node(const NodeID u) {
        switch (_core._options.rating_map_strategy) {
        case RatingMapStrategy::GROWING_HASH_TABLES:
          handle_next_node<RatingMapStrategy::GROWING_HASH_TABLES>(u);
          break;

        case RatingMapStrategy::SINGLE_PHASE:
          handle_next_node<RatingMapStrategy::SINGLE_PHASE>(u);
          break;

        case RatingMapStrategy::TWO_PHASE:
          handle_next_node<RatingMapStrategy::TWO_PHASE>(u);
          break;
        }
      }

      template <RatingMapStrategy Strategy> void handle_next_node(const NodeID u) {
        _current_node = u;
        if (!should_consider(u)) {
          return;
        }

        ++_stats.processed_nodes;

        if constexpr (Strategy == RatingMapStrategy::GROWING_HASH_TABLES) {
          handle_node_with_map(growing_rating_map());
        } else if constexpr (Strategy == RatingMapStrategy::SINGLE_PHASE) {
          handle_node_with_map(rating_map());
        } else if constexpr (Strategy == RatingMapStrategy::TWO_PHASE) {
          handle_first_phase_node();
        }
      }

    private:
      template <typename LocalRatingMap> void handle_node_with_map(LocalRatingMap &rating_map) {
        const BestMove move = _core.find_best_move(
            _current_node, _rand, rating_map, _tie_breaking_clusters, _tie_breaking_favored_clusters
        );
        _core.try_commit_move(move, _stats);
      }

      void handle_first_phase_node() {
        if constexpr (Workspace::kSupportsTwoPhase) {
          const NodeID u = _current_node;
          const NodeWeight u_weight = _core._graph.node_weight(u);
          const ClusterID u_cluster = _core._labels.cluster(u);
          auto &map = rating_map();
          const std::size_t upper_bound_size = std::min<std::size_t>(
              {_core._graph.degree(u),
               _core._initial_num_clusters,
               _core._options.rating_map_threshold}
          );

          const auto maybe_move = map.execute(upper_bound_size, [&](auto &actual_map) {
            return _core.find_best_move_first_phase(
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

      RatingMap &rating_map() {
        if (_rating_map == nullptr) {
          _rating_map = &_core._workspace.rating_map_ets.local();
        }
        return *_rating_map;
      }

      GrowingRatingMap &growing_rating_map() {
        if (_growing_rating_map == nullptr) {
          _growing_rating_map = &_core._workspace.growing_rating_map_ets.local();
        }
        return *_growing_rating_map;
      }

      friend class Pass;

      LabelPropagationCore &_core;
      PassStats &_stats;
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
      if (_core._options.rating_map_strategy == RatingMapStrategy::TWO_PHASE) {
        _core.finish_second_phase(_stats.local());
      }

      Result result;
      for (const PassStats &local_stats : _stats) {
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
    tbb::enumerable_thread_specific<PassStats> _stats;
  };

  [[nodiscard]] Pass begin_pass() {
    _workspace.second_phase_nodes.clear();
    return Pass(*this);
  }

  template <typename LocalRatingMap>
  [[nodiscard]] BestMove find_best_move(
      const NodeID u,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const NodeWeight u_weight = _graph.node_weight(u);
    const ClusterID u_cluster = _labels.cluster(u);
    return find_best_move_impl(
        u,
        u_weight,
        u_cluster,
        rand,
        rating_map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );
  }

  [[nodiscard]] BestMove find_best_move(const NodeID u, Random &rand) {
    auto &rating_map = _workspace.rating_map_ets.local();
    auto &tie_breaking_clusters = _workspace.tie_breaking_clusters_ets.local();
    auto &tie_breaking_favored_clusters = _workspace.tie_breaking_favored_clusters_ets.local();
    return find_best_move(
        u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  std::pair<bool, bool> try_commit_move(const BestMove &move, PassStats &stats) {
    if (!move.valid || _labels.cluster(move.node) == move.new_cluster) {
      return {false, false};
    }

    const bool successful_weight_move = _weights.move_cluster_weight(
        move.old_cluster,
        move.new_cluster,
        move.node_weight,
        _weights.max_cluster_weight(move.new_cluster)
    );

    if (!successful_weight_move) {
      return {false, false};
    }

    _labels.move_node(move.node, move.new_cluster);
    activate_neighbors(move.node);
    stats.expected_total_gain += move.gain;

    const bool emptied_cluster =
        _options.track_cluster_count && _weights.cluster_weight(move.old_cluster) == 0;
    if (emptied_cluster) {
      ++stats.removed_clusters;
    }
    ++stats.moved_nodes;
    return {true, emptied_cluster};
  }

  void activate_neighbors(const NodeID u) {
    if (_options.active_set_strategy == ActiveSetStrategy::NONE) {
      return;
    }

    _graph.adjacent_nodes(u, [&](const NodeID v) {
      if (_neighbors.activate(v) && v < _workspace.active.size()) {
        __atomic_store_n(&_workspace.active[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

  void activate_neighbors_of_ghost_node(const NodeID u) {
    KASSERT(_graph.is_ghost_node(u));
    if (_options.active_set_strategy != ActiveSetStrategy::GLOBAL) {
      return;
    }

    _graph.ghost_graph().adjacent_nodes(u, [&](const NodeID v) {
      if (_neighbors.activate(v) && v < _workspace.active.size()) {
        __atomic_store_n(&_workspace.active[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

#include "kaminpar-common/label_propagation/postprocessing_public.inc"

private:
  [[nodiscard]] bool should_consider(const NodeID u) const {
    if (u >= _num_active_nodes) {
      return false;
    }
    if (_neighbors.skip(u)) {
      return false;
    }
    if (_graph.degree(u) >= _options.max_degree) {
      return false;
    }
    if (_options.active_set_strategy != ActiveSetStrategy::NONE) {
      if (u >= _workspace.active.size() ||
          !__atomic_load_n(&_workspace.active[u], __ATOMIC_RELAXED)) {
        return false;
      }
    }
    return true;
  }

  template <typename LocalRatingMap>
  [[nodiscard]] BestMove find_best_move_impl(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      LocalRatingMap &rating_map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    auto action = [&](auto &map) {
      return compute_best_move_from_map(
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

  template <typename LocalRatingMap>
  [[nodiscard]] std::optional<BestMove> find_best_move_first_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      LocalRatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    bool is_second_phase_node = false;
    bool is_interface_node = false;

    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) -> bool {
      if (_neighbors.accept(u, v)) {
        const ClusterID v_cluster = _labels.cluster(v);
        map[v_cluster] += w;

        if (map.size() >= _options.rating_map_threshold) [[unlikely]] {
          is_second_phase_node = true;
          return true;
        }

        if (_options.active_set_strategy == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }

      return false;
    };

    if (_options.max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _options.max_num_neighbors, add_to_rating_map);
    }

    if (is_second_phase_node) [[unlikely]] {
      map.clear();
      _workspace.second_phase_nodes.push_back(u);
      return std::nullopt;
    }

    clear_active(u, is_interface_node);

    return compute_best_move_after_rating(
        u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  template <typename ActualMap>
  [[nodiscard]] BestMove compute_best_move_from_map(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    bool is_interface_node = false;

    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) {
      if (_neighbors.accept(u, v)) {
        const ClusterID v_cluster = _labels.cluster(v);
        map[v_cluster] += w;

        if (_options.active_set_strategy == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }
    };

    if (_options.max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _options.max_num_neighbors, add_to_rating_map);
    }

    clear_active(u, is_interface_node);

    return compute_best_move_after_rating(
        u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  template <typename ActualMap>
  [[nodiscard]] BestMove compute_best_move_after_rating(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = _weights.cluster_weight(u_cluster);
    SelectionState state{
        .local_rand = rand,
        .u = u,
        .u_weight = u_weight,
        .initial_cluster = u_cluster,
        .initial_cluster_weight = initial_cluster_weight,
        .best_cluster = u_cluster,
        .best_gain = 0,
        .best_cluster_weight = initial_cluster_weight,
        .overall_best_gain = 0,
        .current_cluster = 0,
        .current_gain = 0,
        .current_cluster_weight = 0,
    };

    const bool store_favored_cluster =
        _options.use_two_hop_clustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _options.use_actual_gain ? map[u_cluster] : 0;
    const ClusterID favored_cluster = _selector.select(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );

    if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
      _workspace.favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = state.best_gain - map[state.initial_cluster];
    map.clear();
    return {
        .node = u,
        .node_weight = u_weight,
        .old_cluster = u_cluster,
        .new_cluster = state.best_cluster,
        .gain = actual_gain,
        .valid = true,
    };
  }

  void clear_active(const NodeID u, const bool is_interface_node) {
    if (_options.active_set_strategy == ActiveSetStrategy::GLOBAL) {
      __atomic_store_n(&_workspace.active[u], 0, __ATOMIC_RELAXED);
    } else if (_options.active_set_strategy == ActiveSetStrategy::LOCAL && !is_interface_node) {
      __atomic_store_n(&_workspace.active[u], 0, __ATOMIC_RELAXED);
    }
  }

  void finish_second_phase(PassStats &stats) {
    if constexpr (Workspace::kSupportsTwoPhase) {
      const std::size_t num_clusters = _initial_num_clusters;
      if (_workspace.concurrent_rating_map.capacity() < num_clusters) {
        _workspace.concurrent_rating_map.resize(num_clusters);
      }

      if (!_workspace.second_phase_nodes.empty() && _options.relabel_before_second_phase) {
        relabel_clusters();
      }

      auto &rand = Random::instance();
      for (const NodeID u : _workspace.second_phase_nodes) {
        if (_neighbors.skip(u)) {
          continue;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        const ClusterID u_cluster = _labels.cluster(u);
        const BestMove move = find_best_move_second_phase(
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

  [[nodiscard]] BestMove find_best_move_second_phase(
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
        u, _options.max_num_neighbors, 2000, [&](auto &&pfor_adjacent_nodes) {
          auto &local_used_entries = map.local_used_entries();
          auto &local_rating_map = _workspace.rating_map_ets.local().small_map();

          pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight w) {
            if (_neighbors.accept(u, v)) {
              const ClusterID v_cluster = _labels.cluster(v);
              local_rating_map[v_cluster] += w;

              if (local_rating_map.size() >= _options.rating_map_threshold) [[unlikely]] {
                flush_local_rating_map(local_used_entries, local_rating_map);
              }

              if (_options.active_set_strategy == ActiveSetStrategy::LOCAL) {
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

    const bool store_favored_cluster =
        _options.use_two_hop_clustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _options.use_actual_gain ? map[u_cluster] : 0;

    map.iterate_and_reset([&](const auto i, auto &local_entries) {
      SelectionState local_state{
          .local_rand = Random::instance(),
          .u = u,
          .u_weight = u_weight,
          .initial_cluster = u_cluster,
          .initial_cluster_weight = initial_cluster_weight,
          .best_cluster = u_cluster,
          .best_gain = 0,
          .best_cluster_weight = initial_cluster_weight,
          .overall_best_gain = 0,
          .current_cluster = 0,
          .current_gain = 0,
          .current_cluster_weight = 0,
      };

      const ClusterID local_favored_cluster = _selector.select(
          store_favored_cluster,
          gain_delta,
          local_state,
          local_entries,
          _workspace.tie_breaking_clusters_ets.local(),
          _workspace.tie_breaking_favored_clusters_ets.local()
      );
      const EdgeWeight local_favored_cluster_gain = map[local_favored_cluster];

      _workspace.local_cluster_selection_states[i] = {
          local_state.best_gain,
          local_state.best_cluster,
          local_favored_cluster_gain,
          local_favored_cluster,
      };
    });

    ClusterID favored_cluster = u_cluster;
    ClusterID best_cluster = u_cluster;
    EdgeWeight best_gain = 0;

    if (_options.tie_breaking_strategy == TieBreakingStrategy::UNIFORM) {
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

        if (store_favored_cluster) {
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

        if (store_favored_cluster && local_state.favored_cluster_gain > favored_cluster_gain) {
          favored_cluster_gain = local_state.favored_cluster_gain;
          favored_cluster = local_state.favored_cluster;
        }

        local_state.best_gain = -1;
        local_state.favored_cluster_gain = -1;
      }
    }

    if (store_favored_cluster && best_cluster == u_cluster) {
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
  Options<NodeID, ClusterID> _options;

  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
  ClusterID _prev_num_clusters = 0;
  ClusterID _initial_num_clusters = 0;
  parallel::Atomic<ClusterID> _current_num_clusters = 0;
  parallel::Atomic<EdgeWeight> _expected_total_gain = 0;
  bool _relabeled = false;
};

template <RatingMapStrategy Strategy, typename Order, typename Core, typename Pass>
void run_iteration_with_strategy(Order &order, Core &lp, Pass &pass) {
  auto make_local = [&] {
    return pass.local();
  };
  auto handle_node = [](auto &local, const auto node) {
    local.template handle_next_node<Strategy>(node);
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
      local.template handle_next_node<Strategy>(node);
    });
  }
}

template <typename Order, typename Core>
typename Core::Result run_iteration(Order &order, Core &lp) {
  auto pass = lp.begin_pass();

  switch (lp.options().rating_map_strategy) {
  case RatingMapStrategy::GROWING_HASH_TABLES:
    run_iteration_with_strategy<RatingMapStrategy::GROWING_HASH_TABLES>(order, lp, pass);
    break;

  case RatingMapStrategy::SINGLE_PHASE:
    run_iteration_with_strategy<RatingMapStrategy::SINGLE_PHASE>(order, lp, pass);
    break;

  case RatingMapStrategy::TWO_PHASE:
    run_iteration_with_strategy<RatingMapStrategy::TWO_PHASE>(order, lp, pass);
    break;
  }

  return pass.finish();
}

} // namespace kaminpar::lp
