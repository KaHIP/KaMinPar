/*******************************************************************************
 * Per-node label propagation processing engine.
 *
 * Handles the core LP logic for a single node: fill the rating map with
 * neighboring cluster weights, invoke the selection strategy to pick the best
 * cluster, attempt the move, and activate neighbors if the move succeeds.
 *
 * This is a standalone template — no CRTP. All dependencies are injected as
 * template parameters and member references.
 *
 * @file:   node_processor.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <limits>
#include <optional>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/label_propagation/active_set.h"
#include "kaminpar-shm/label_propagation/config.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::lp {

/*!
 * Per-node LP processing engine.
 *
 * @tparam Graph The graph type.
 * @tparam ClusterOps Provides cluster/weight queries (cluster, move_node, cluster_weight, etc.).
 * @tparam Selection The cluster selection strategy (SimpleGainClusterSelection or
 *         OverloadAwareClusterSelection).
 * @tparam Config The LP config type (derives from LabelPropagationConfig).
 */
template <typename Graph, typename ClusterOps, typename Selection, typename Config>
class LPNodeProcessor {
  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;

  using ClusterID = typename Config::ClusterID;
  using ClusterWeight = typename Config::ClusterWeight;

  using RatingMap = typename Config::RatingMap;
  using ConcurrentRatingMap = typename Config::ConcurrentRatingMap;
  using GrowingRatingMap = typename Config::GrowingRatingMap;

  using SelectionState = ClusterSelectionState<ClusterID, ClusterWeight, NodeWeight, EdgeWeight>;
  using LocalSelectionState = LocalClusterSelectionState<ClusterID, EdgeWeight>;

  static constexpr bool kUseActiveSet =
      Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy;

public:
  //! Data structures that can be saved/restored for memory reuse between LP runs.
  using DataStructures = std::tuple<
      tbb::enumerable_thread_specific<RatingMap>,
      tbb::enumerable_thread_specific<GrowingRatingMap>,
      ConcurrentRatingMap,
      tbb::enumerable_thread_specific<std::vector<ClusterID>>,
      tbb::enumerable_thread_specific<std::vector<ClusterID>>,
      StaticArray<std::uint8_t>,
      StaticArray<std::uint8_t>,
      StaticArray<ClusterID>,
      tbb::concurrent_vector<NodeID>>;

  LPNodeProcessor(
      ClusterOps &ops,
      Selection &selection,
      ActiveSet<kUseActiveSet> &active_set
  )
      : _ops(ops),
        _selection(selection),
        _active_set(active_set) {}

  void setup(DataStructures structs) {
    auto
        [rating_map_ets,
         growing_rating_map_ets,
         concurrent_rating_map,
         tie_breaking_clusters_ets,
         tie_breaking_favored_clusters_ets,
         active,
         moved,
         favored_clusters,
         second_phase_nodes] = std::move(structs);
    _rating_map_ets = std::move(rating_map_ets);
    _growing_rating_map_ets = std::move(growing_rating_map_ets);
    _concurrent_rating_map = std::move(concurrent_rating_map);
    _tie_breaking_clusters_ets = std::move(tie_breaking_clusters_ets);
    _tie_breaking_favored_clusters_ets = std::move(tie_breaking_favored_clusters_ets);
    _active_set.set(std::move(active));
    _moved = std::move(moved);
    _favored_clusters = std::move(favored_clusters);
    _second_phase_nodes = std::move(second_phase_nodes);
  }

  DataStructures release() {
    return std::make_tuple(
        std::move(_rating_map_ets),
        std::move(_growing_rating_map_ets),
        std::move(_concurrent_rating_map),
        std::move(_tie_breaking_clusters_ets),
        std::move(_tie_breaking_favored_clusters_ets),
        _active_set.take(),
        std::move(_moved),
        std::move(_favored_clusters),
        std::move(_second_phase_nodes)
    );
  }

  /*!
   * Allocate data structures for a graph with `num_nodes` nodes and
   * `num_clusters` initial clusters.
   */
  void allocate(const NodeID num_nodes, const NodeID num_active_nodes, const ClusterID num_clusters) {
    _num_nodes = num_nodes;
    _num_active_nodes = num_active_nodes;

    _active_set.allocate(num_active_nodes);

    if constexpr (Config::kUseTwoHopClustering) {
      if (_favored_clusters.size() < num_active_nodes) {
        _favored_clusters.resize(num_active_nodes);
      }
    }

    if (_rating_map_ets.empty() || _prev_num_clusters < num_clusters) {
      _rating_map_ets =
          tbb::enumerable_thread_specific<RatingMap>([num_clusters] {
            return RatingMap(num_clusters);
          });
    } else {
      for (auto &rating_map : _rating_map_ets) {
        rating_map.change_max_size(num_clusters);
      }
    }

    _prev_num_clusters = num_clusters;
  }

  void free() {
    _rating_map_ets.clear();
    _growing_rating_map_ets.clear();
    _tie_breaking_clusters_ets.clear();
    _tie_breaking_favored_clusters_ets.clear();
    _prev_num_clusters = 0;

    _active_set.free();
    _favored_clusters.free();
    _moved.free();

    _second_phase_nodes.clear();
    _second_phase_nodes.shrink_to_fit();

    _concurrent_rating_map.free();
  }

  /*!
   * Initialize state for a new LP run.
   */
  void initialize(const Graph *graph, const ClusterID num_clusters) {
    _graph = graph;
    _initial_num_clusters = num_clusters;
    _current_num_clusters = num_clusters;
    _local_cluster_selection_states.resize(
        tbb::this_task_arena::max_concurrency(), {-1, 0, -1, 0}
    );
    reset_state();
  }

  /*!
   * Process a single node: fill rating map -> select best cluster -> try move.
   * @return {moved, emptied_cluster}
   */
  template <typename RatingMapT>
  std::pair<bool, bool> handle_node(
      const NodeID u,
      Random &rand,
      RatingMapT &map,
      std::vector<ClusterID> &tie_breaking_clusters,
      std::vector<ClusterID> &tie_breaking_favored_clusters
  ) {
    if (_ops.skip_node(u)) {
      return {false, false};
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = _ops.cluster(u);

    const auto [best_cluster, gain] = [&] {
      if constexpr (std::is_same_v<RatingMapT, GrowingRatingMap>) {
        return find_best_cluster(
            u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
        );
      } else {
        const std::size_t upper_bound_size =
            std::min<ClusterID>(_graph->degree(u), _initial_num_clusters);
        return map.execute(upper_bound_size, [&](auto &actual_map) {
          return find_best_cluster(
              u,
              u_weight,
              u_cluster,
              rand,
              actual_map,
              tie_breaking_clusters,
              tie_breaking_favored_clusters
          );
        });
      }
    }();

    return try_node_move(u, u_weight, u_cluster, best_cluster, gain);
  }

  /*!
   * First-phase node processing: may defer high-degree nodes to second phase.
   */
  std::pair<bool, bool> handle_first_phase_node(
      const NodeID u,
      Random &rand,
      RatingMap &map,
      std::vector<ClusterID> &tie_breaking_clusters,
      std::vector<ClusterID> &tie_breaking_favored_clusters
  ) {
    if (!_ops.skip_node(u)) {
      const NodeWeight u_weight = _graph->node_weight(u);
      const ClusterID u_cluster = _ops.cluster(u);

      const std::size_t upper_bound_size = std::min<std::size_t>(
          {_graph->degree(u), _initial_num_clusters, Config::kRatingMapThreshold}
      );

      const auto maybe_move = map.execute(upper_bound_size, [&](auto &actual_map) {
        return find_best_cluster_first_phase(
            u,
            u_weight,
            u_cluster,
            rand,
            actual_map,
            tie_breaking_clusters,
            tie_breaking_favored_clusters
        );
      });

      if (maybe_move.has_value()) {
        const auto [best_cluster, gain] = *maybe_move;
        return try_node_move(u, u_weight, u_cluster, best_cluster, gain);
      }
    }

    return {false, false};
  }

  /*!
   * Second-phase node processing: uses concurrent rating map for high-degree nodes.
   */
  std::pair<bool, bool>
  handle_second_phase_node(const NodeID u, Random &rand, ConcurrentRatingMap &map) {
    if (_ops.skip_node(u)) {
      return {false, false};
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = _ops.cluster(u);

    const auto [best_cluster, gain] =
        find_best_cluster_second_phase(u, u_weight, u_cluster, rand, map);
    return try_node_move<true>(u, u_weight, u_cluster, best_cluster, gain);
  }

  // --- Accessors ---

  [[nodiscard]] bool should_stop() const {
    if constexpr (Config::kTrackClusterCount) {
      return _current_num_clusters <= _desired_num_clusters;
    }
    return false;
  }

  [[nodiscard]] bool is_active(const NodeID u) const {
    return _active_set.is_active(u);
  }

  void set_desired_num_clusters(const ClusterID desired) {
    _desired_num_clusters = desired;
  }

  void set_max_num_neighbors(const NodeID max) {
    _max_num_neighbors = max;
  }

  [[nodiscard]] parallel::Atomic<ClusterID> &current_num_clusters() {
    return _current_num_clusters;
  }

  [[nodiscard]] const parallel::Atomic<ClusterID> &current_num_clusters() const {
    return _current_num_clusters;
  }

  [[nodiscard]] ClusterID initial_num_clusters() const {
    return _initial_num_clusters;
  }

  [[nodiscard]] StaticArray<ClusterID> &favored_clusters() {
    return _favored_clusters;
  }

  [[nodiscard]] StaticArray<std::uint8_t> &moved() {
    return _moved;
  }

  [[nodiscard]] bool relabeled() const {
    return _relabeled;
  }

  void set_relabeled(const bool relabeled) {
    _relabeled = relabeled;
  }

  void set_initial_num_clusters(const ClusterID n) {
    _initial_num_clusters = n;
  }

  [[nodiscard]] tbb::concurrent_vector<NodeID> &second_phase_nodes() {
    return _second_phase_nodes;
  }

  [[nodiscard]] ConcurrentRatingMap &concurrent_rating_map() {
    return _concurrent_rating_map;
  }

  [[nodiscard]] tbb::enumerable_thread_specific<RatingMap> &rating_map_ets() {
    return _rating_map_ets;
  }

  [[nodiscard]] tbb::enumerable_thread_specific<std::vector<ClusterID>> &tie_breaking_clusters_ets() {
    return _tie_breaking_clusters_ets;
  }

  [[nodiscard]] tbb::enumerable_thread_specific<std::vector<ClusterID>> &tie_breaking_favored_clusters_ets() {
    return _tie_breaking_favored_clusters_ets;
  }

  [[nodiscard]] EdgeWeight expected_total_gain() const {
    return _expected_total_gain;
  }

  /*!
   * Perform relabeling: compact cluster IDs to [0, num_actual_clusters).
   * Updates favored_clusters and moved arrays as needed.
   */
  void relabel_clusters() {
    SCOPED_HEAP_PROFILER("Relabel");
    SCOPED_TIMER("Relabel");

    ClusterID num_actual_clusters = _current_num_clusters;
    _initial_num_clusters = num_actual_clusters;
    _relabeled = true;

    if constexpr (Config::kUseTwoHopClustering) {
      if (_moved.size() < _graph->n()) {
        _moved.resize(_graph->n());
      }
    }

    StaticArray<ClusterID> mapping(_graph->n());
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph->n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        const ClusterID c_u = _ops.cluster(u);
        __atomic_store_n(&mapping[c_u], 1, __ATOMIC_RELAXED);

        if constexpr (Config::kUseTwoHopClustering) {
          if (u != c_u) {
            _moved[u] = 1;
          }
        }
      }
    });

    parallel::prefix_sum(mapping.begin(), mapping.end(), mapping.begin());
    KASSERT(num_actual_clusters == mapping[_graph->n() - 1]);

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph->n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        _ops.move_node(u, mapping[_ops.cluster(u)] - 1);
        _favored_clusters[u] = mapping[_favored_clusters[u]] - 1;
      }
    });

    _ops.reassign_cluster_weights(mapping, num_actual_clusters);
  }

private:
  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
            _active_set.mark_active(u);

            const ClusterID initial_cluster = _ops.initial_cluster(u);
            _ops.init_cluster(u, initial_cluster);
            if constexpr (Config::kUseTwoHopClustering) {
              _favored_clusters[u] = initial_cluster;
            }

            _ops.reset_node_state(u);
          });
        },
        [&] {
          tbb::parallel_for<ClusterID>(0, _initial_num_clusters, [&](const ClusterID cluster) {
            _ops.init_cluster_weight(cluster, _ops.initial_cluster_weight(cluster));
          });
        }
    );
    IFSTATS(_expected_total_gain = 0);
    _current_num_clusters = _initial_num_clusters;
    _relabeled = false;
  }

  template <typename RatingMapT>
  std::pair<ClusterID, EdgeWeight> find_best_cluster(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      RatingMapT &map,
      std::vector<ClusterID> &tie_breaking_clusters,
      std::vector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = _ops.cluster_weight(u_cluster);
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

    bool is_interface_node = false;
    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) {
      if (_ops.accept_neighbor(u, v)) {
        const ClusterID v_cluster = _ops.cluster(v);
        map[v_cluster] += w;

        if constexpr (Config::kUseLocalActiveSetStrategy) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }
    };

    if (_max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph->adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph->adjacent_nodes(u, _max_num_neighbors, add_to_rating_map);
    }

    if constexpr (Config::kUseActiveSetStrategy) {
      _active_set.mark_inactive(u);
    } else if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (!is_interface_node) {
        _active_set.mark_inactive(u);
      }
    }

    const bool store_favored_cluster =
        Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _ops.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;
    ClusterID favored_cluster = _selection.select_best_cluster(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );

    if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
      _favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = IFSTATS(state.best_gain - map[state.initial_cluster]);
    map.clear();
    return std::make_pair(state.best_cluster, actual_gain);
  }

  template <typename RatingMapT>
  std::optional<std::pair<ClusterID, EdgeWeight>> find_best_cluster_first_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      RatingMapT &map,
      std::vector<ClusterID> &tie_breaking_clusters,
      std::vector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = _ops.cluster_weight(u_cluster);
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

    bool is_interface_node = false;
    bool is_second_phase_node = false;
    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) -> bool {
      if (_ops.accept_neighbor(u, v)) {
        const ClusterID v_cluster = _ops.cluster(v);
        map[v_cluster] += w;

        if (map.size() >= Config::kRatingMapThreshold) [[unlikely]] {
          is_second_phase_node = true;
          return true;
        }

        if constexpr (Config::kUseLocalActiveSetStrategy) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }

      return false;
    };

    if (_max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph->adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph->adjacent_nodes(u, _max_num_neighbors, add_to_rating_map);
    }

    if (is_second_phase_node) [[unlikely]] {
      map.clear();
      _second_phase_nodes.push_back(u);
      return std::nullopt;
    }

    if constexpr (Config::kUseActiveSetStrategy) {
      _active_set.mark_inactive(u);
    } else if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (!is_interface_node) {
        _active_set.mark_inactive(u);
      }
    }

    const bool store_favored_cluster =
        Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _ops.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;
    ClusterID favored_cluster = _selection.select_best_cluster(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );

    if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
      _favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = IFSTATS(state.best_gain - map[state.initial_cluster]);
    map.clear();
    return std::make_pair(state.best_cluster, actual_gain);
  }

  std::pair<ClusterID, EdgeWeight> find_best_cluster_second_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ConcurrentRatingMap &map
  ) {
    const ClusterWeight initial_cluster_weight = _ops.cluster_weight(u_cluster);

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
    _graph->pfor_adjacent_nodes(u, _max_num_neighbors, 2000, [&](auto &&pfor_adjacent_nodes) {
      auto &local_used_entries = map.local_used_entries();
      auto &local_rating_map = _rating_map_ets.local().small_map();

      pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight w) {
        if (_ops.accept_neighbor(u, v)) {
          const ClusterID v_cluster = _ops.cluster(v);
          local_rating_map[v_cluster] += w;

          if (local_rating_map.size() >= Config::kRatingMapThreshold) [[unlikely]] {
            flush_local_rating_map(local_used_entries, local_rating_map);
          }

          if constexpr (Config::kUseLocalActiveSetStrategy) {
            is_interface_node |= v >= _num_active_nodes;
          }
        }
      });
    });

    tbb::parallel_for(_rating_map_ets.range(), [&](auto &rating_maps) {
      auto &local_used_entries = map.local_used_entries();
      for (auto &rating_map : rating_maps) {
        auto &local_rating_map = rating_map.small_map();
        flush_local_rating_map(local_used_entries, local_rating_map);
      }
    });

    if constexpr (Config::kUseActiveSetStrategy) {
      _active_set.mark_inactive(u);
    } else if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (!is_interface_node) {
        _active_set.mark_inactive(u);
      }
    }

    const bool store_favored_cluster =
        Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _ops.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;

    map.iterate_and_reset([&](const auto i, const auto &local_entries) {
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

      const ClusterID local_favored_cluster = _selection.select_best_cluster(
          store_favored_cluster,
          gain_delta,
          local_state,
          local_entries,
          _tie_breaking_clusters_ets.local(),
          _tie_breaking_favored_clusters_ets.local()
      );
      const EdgeWeight local_favored_cluster_gain = map[local_favored_cluster];

      _local_cluster_selection_states[i] = {
          local_state.best_gain,
          local_state.best_cluster,
          local_favored_cluster_gain,
          local_favored_cluster,
      };
    });

    ClusterID favored_cluster = u_cluster;
    ClusterID best_cluster = u_cluster;
    EdgeWeight best_gain = 0;

    const bool use_uniform_tie_breaking =
        _tie_breaking_strategy == shm::TieBreakingStrategy::UNIFORM;
    if (use_uniform_tie_breaking) {
      auto &tie_breaking_clusters = _tie_breaking_clusters_ets.local();
      auto &tie_breaking_favored_clusters = _tie_breaking_favored_clusters_ets.local();

      EdgeWeight favored_cluster_gain = 0;
      for (LocalSelectionState &local_state : _local_cluster_selection_states) {
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
      for (LocalSelectionState &local_state : _local_cluster_selection_states) {
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
      _favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = IFSTATS(best_gain - map[u_cluster]);
    return std::make_pair(best_cluster, actual_gain);
  }

  template <bool kParallelActivate = false>
  std::pair<bool, bool> try_node_move(
      NodeID u, NodeWeight u_weight, ClusterID u_cluster, ClusterID new_cluster, EdgeWeight gain
  ) {
    if (_ops.cluster(u) != new_cluster) {
      const bool successful_weight_move = _ops.move_cluster_weight(
          u_cluster, new_cluster, u_weight, _ops.max_cluster_weight(new_cluster)
      );

      if (successful_weight_move) {
        _ops.move_node(u, new_cluster);
        activate_neighbors<kParallelActivate>(u);
        IFSTATS(_expected_total_gain += gain);

        const bool decrement_cluster_count =
            Config::kTrackClusterCount && _ops.cluster_weight(u_cluster) == 0;
        return {true, decrement_cluster_count};
      }
    }

    return {false, false};
  }

  template <bool kParallel = false> void activate_neighbors(const NodeID u) {
    const auto activate = [&](const NodeID v) {
      if (_ops.activate_neighbor(v)) {
        _active_set.mark_active(v);
      }
    };

    if constexpr (kParallel) {
      _graph->pfor_adjacent_nodes(
          u,
          std::numeric_limits<NodeID>::max(),
          20000,
          [&](const NodeID v, [[maybe_unused]] const EdgeWeight) { activate(v); }
      );
    } else {
      _graph->adjacent_nodes(u, activate);
    }
  }

  // --- Members ---

  const Graph *_graph = nullptr;
  ClusterOps &_ops;
  Selection &_selection;
  ActiveSet<kUseActiveSet> &_active_set;

  ClusterID _initial_num_clusters = 0;
  parallel::Atomic<ClusterID> _current_num_clusters = 0;
  ClusterID _desired_num_clusters = 0;

  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _prev_num_clusters = 0;

  NodeID _max_num_neighbors = std::numeric_limits<NodeID>::max();

  shm::TieBreakingStrategy _tie_breaking_strategy = shm::TieBreakingStrategy::GEOMETRIC;

  tbb::enumerable_thread_specific<RatingMap> _rating_map_ets;
  tbb::enumerable_thread_specific<GrowingRatingMap> _growing_rating_map_ets;
  ConcurrentRatingMap _concurrent_rating_map;

  tbb::enumerable_thread_specific<std::vector<ClusterID>> _tie_breaking_clusters_ets;
  tbb::enumerable_thread_specific<std::vector<ClusterID>> _tie_breaking_favored_clusters_ets;

  CacheAlignedVector<LocalSelectionState> _local_cluster_selection_states;

  StaticArray<std::uint8_t> _moved;
  bool _relabeled = false;
  StaticArray<ClusterID> _favored_clusters;

  tbb::concurrent_vector<NodeID> _second_phase_nodes;

  parallel::Atomic<EdgeWeight> _expected_total_gain = 0;
};

} // namespace kaminpar::lp
