/*******************************************************************************
 * Generic implementation of parallel label propagation.
 *
 * @file:   label_propagation.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <atomic>
#include <limits>
#include <optional>
#include <type_traits>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar {

struct LabelPropagationConfig {
  // Data structures used to accumulate edge weights for gain value calculation
  using RatingMap = ::kaminpar::RatingMap<shm::EdgeWeight, shm::NodeID>;
  using ConcurrentRatingMap = ConcurrentFastResetArray<shm::EdgeWeight, shm::NodeID>;
  using GrowingRatingMap = DynamicRememberingFlatMap<shm::NodeID, shm::EdgeWeight>;

  // Data type for cluster IDs and weights
  using ClusterID = void;
  using ClusterWeight = void;

  // Approx. number of edges per work unit
  static constexpr shm::NodeID kMinChunkSize = 1024;

  // Nodes per permutation unit: when iterating over nodes in a chunk, we divide
  // them into permutation units, iterate over permutation orders in random
  // order, and iterate over nodes inside a permutation unit in random order.
  static constexpr shm::NodeID kPermutationSize = 64;

  // When randomizing the node order inside a permutation unit, we pick a random
  // permutation from a pool of permutations. This constant determines the pool
  // size.
  static constexpr std::size_t kNumberOfNodePermutations = 64;

  // When computing a new cluster for each node in an iteration, nodes that use more than or equal
  // to the threshold amount of entires in the rating map are processed in a second phase
  // sequentially.
  static constexpr std::size_t kRatingMapThreshold = 10000;

  // If true, we count the number of empty clusters
  static constexpr bool kTrackClusterCount = true;

  // If true, match singleton clusters in 2-hop distance
  static constexpr bool kUseTwoHopClustering = false;

  static constexpr bool kUseActualGain = false;

  static constexpr bool kUseActiveSetStrategy = true;
  static constexpr bool kUseLocalActiveSetStrategy = false;
};

/*!
 * Generic implementation of parallel label propagation. To use, inherit from
 * this class and implement all mandatory template functions.
 *
 * @tparam Derived Derived class for static polymorphism.
 * @tparam Config Algorithmic configuration and data types.
 */
template <typename Derived, typename Config, typename Graph> class LabelPropagation {
  static_assert(std::is_base_of_v<LabelPropagationConfig, Config>);

  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

protected:
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;

  using ClusterID = typename Config::ClusterID;
  using ClusterWeight = typename Config::ClusterWeight;

  using RatingMap = typename Config::RatingMap;
  using ConcurrentRatingMap = typename Config::ConcurrentRatingMap;
  using GrowingRatingMap = typename Config::GrowingRatingMap;

  using LabelPropagationImplementation = shm::LabelPropagationImplementation;
  using TieBreakingStrategy = shm::TieBreakingStrategy;

public:
  void set_max_degree(const NodeID max_degree) {
    _max_degree = max_degree;
  }
  [[nodiscard]] NodeID max_degree() const {
    return _max_degree;
  }

  void set_max_num_neighbors(const NodeID max_num_neighbors) {
    _max_num_neighbors = max_num_neighbors;
  }
  [[nodiscard]] NodeID max_num_neighbors() const {
    return _max_num_neighbors;
  }

  void set_desired_num_clusters(const ClusterID desired_num_clusters) {
    _desired_num_clusters = desired_num_clusters;
  }
  [[nodiscard]] ClusterID desired_num_clusters() const {
    return _desired_num_clusters;
  }

  void set_implementation(const LabelPropagationImplementation impl) {
    _impl = impl;
  }
  [[nodiscard]] LabelPropagationImplementation implementation() {
    return _impl;
  }

  void set_tie_breaking_strategy(const TieBreakingStrategy strategy) {
    _tie_breaking_strategy = strategy;
  }
  [[nodiscard]] TieBreakingStrategy tie_breaking_strategy() {
    return _tie_breaking_strategy;
  }

  void set_relabel_before_second_phase(const bool relabel) {
    _relabel_before_second_phase = relabel;
  }
  [[nodiscard]] bool relabel_before_second_phase() const {
    return _relabel_before_second_phase;
  }

  [[nodiscard]] EdgeWeight expected_total_gain() const {
    return _expected_total_gain;
  }

protected:
  /*!
   * Selects the number of nodes \c num_nodes of the graph for which a clustering is to be
   * computed and the number of clusters \c num_clusters.
   *
   * @param num_nodes Number of nodes in the graph.
   * @param num_clusters The number of clusters.
   */
  void preinitialize(const NodeID num_nodes, const ClusterID num_clusters) {
    preinitialize(num_nodes, num_nodes, num_clusters);
  }

  /*!
   * Selects the number of nodes \c num_nodes of the graph for which a clustering is to be
   * computed, but a clustering is only computed for the first \c num_active_nodes nodes, and the
   * number of clusters \c num_clusters.
   *
   * This is mostly useful for distributed graphs where ghost nodes are always inactive.
   *
   * @param num_nodes Number of nodes in the graph.
   * @param num_active_nodes Number of nodes for which a cluster label is computed.
   * @param num_clusters The number of clusters.
   */
  void preinitialize(
      const NodeID num_nodes, const NodeID num_active_nodes, const ClusterID num_clusters
  ) {
    _num_nodes = num_nodes;
    _num_active_nodes = num_active_nodes;
    _prev_num_clusters = _num_clusters;
    _num_clusters = num_clusters;
  }

  /*!
   * (Re)allocates memory to run label propagation on. Must be called after \c preinitialize().
   */
  void allocate() {
    if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (_active.size() < _num_nodes) {
        _active.resize(_num_nodes);
      }
    }

    if constexpr (Config::kUseActiveSetStrategy) {
      if (_active.size() < _num_active_nodes) {
        _active.resize(_num_active_nodes);
      }
    }

    if constexpr (Config::kUseTwoHopClustering) {
      if (_favored_clusters.size() < _num_active_nodes) {
        _favored_clusters.resize(_num_active_nodes);
      }
    }

    if (_rating_map_ets.empty() || _prev_num_clusters < _num_clusters) {
      _rating_map_ets =
          tbb::enumerable_thread_specific<RatingMap>([&_num_clusters = _num_clusters] {
            return RatingMap(_num_clusters);
          });
    } else {
      for (auto &rating_map : _rating_map_ets) {
        rating_map.change_max_size(_num_clusters);
      }
    }
  }

  void free() {
    // No shrink-to-fit call is needed (and provided by the ets-interface) since the clear already
    // frees the memory.
    _rating_map_ets.clear();
    _growing_rating_map_ets.clear();
    _tie_breaking_clusters_ets.clear();
    _prev_num_clusters = 0;

    _active.free();
    _favored_clusters.free();
    _moved.free();

    _second_phase_nodes.clear();
    _second_phase_nodes.shrink_to_fit();
  }

  /*!
   * Initialize label propagation. Must be called after \c allocate().
   * @param graph Graph for label propagation.
   * @param num_clusters Number of different clusters the nodes are placed in
   * initially. When using label propagation as refinement graphutils, this is
   * usually the number of blocks. When using as for clustering, it is usually
   * the number of nodes.
   */
  void initialize(const Graph *graph, const ClusterID num_clusters) {
    KASSERT(
        graph->n() == 0u || (_num_nodes > 0u && _num_active_nodes > 0u),
        "you must call allocate() before initialize()"
    );

    _graph = graph;
    _initial_num_clusters = num_clusters;
    _current_num_clusters = num_clusters;
    _local_cluster_selection_states.resize(tbb::this_task_arena::max_concurrency(), {-1, 0, -1, 0});
    reset_state();
  }

  /*!
   * Determines whether we should stop label propagation because the number of
   * non-empty clusters has been reduced sufficiently.
   * @return Whether label propagation should be stopped now.
   */
  bool should_stop() {
    if (Config::kTrackClusterCount) {
      return _current_num_clusters <= _desired_num_clusters;
    }
    return false;
  }

  /*!
   * Relabel the clusters such that afterwards the cluster IDs are consecutive in the range of [0,
   * num_actual_clusters]; num_actual_clusters is thereby the number of clusters that have at least
   * one member.
   */
  void relabel_clusters() {
    SCOPED_HEAP_PROFILER("Relabel");
    SCOPED_TIMER("Relabel");

    // Update initial num clusters since the maximum cluster ID is now different.
    ClusterID num_actual_clusters = _current_num_clusters;
    _initial_num_clusters = num_actual_clusters;
    _relabeled = true;

    // Store for each node whether it joined another cluster as this information gets lost. This
    // information is needed only by 2-hop clustering.
    if constexpr (Config::kUseTwoHopClustering) {
      if (_moved.size() < _graph->n()) {
        _moved.resize(_graph->n());
      }
    }

    // Compute a mapping from old cluster IDs to new cluster IDs.
    RECORD("mapping") StaticArray<ClusterID> mapping(_graph->n());
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph->n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        const ClusterID c_u = derived_cluster(u);
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
        // Relabel the cluster stored for each node.
        derived_move_node(u, mapping[derived_cluster(u)] - 1);

        // Relabel the clusters stored in the favored clusters vector.
        _favored_clusters[u] = mapping[_favored_clusters[u]] - 1;
      }
    });

    // Reassign the clusters weights such that they match the new cluster IDs.
    derived_reassign_cluster_weights(mapping, num_actual_clusters);
  }

  /*!
   * Move a single node to a new cluster.
   *
   * @param u The node that is moved.
   * @param rand (Thread-local) \c Random object.
   * @param rating_map (Thread-local) Rating map for gain computation.
   * @return Pair with: whether the node was moved to another cluster, whether
   * the previous cluster is now empty.
   */
  template <typename RatingMap>
  std::pair<bool, bool> handle_node(
      const NodeID u,
      Random &rand,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    if (derived_skip_node(u)) {
      return {false, false};
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = derived_cluster(u);

    const auto [best_cluster, gain] = [&] {
      if constexpr (std::is_same_v<RatingMap, GrowingRatingMap>) {
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
   * Move a single node to a new cluster.
   *
   * @param u The node that is moved.
   * @param rand (Thread-local) \c Random object.
   * @param rating_map (Thread-local) Rating map for gain computation.
   * @return Pair with: whether the node was moved to another cluster, whether
   * the previous cluster is now empty.
   */
  std::pair<bool, bool> handle_first_phase_node(
      const NodeID u,
      Random &rand,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    if (!derived_skip_node(u)) {
      const NodeWeight u_weight = _graph->node_weight(u);
      const ClusterID u_cluster = derived_cluster(u);

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
   * Move a single node to a new cluster.
   *
   * @param u The node that is moved.
   * @param rand (Thread-local) \c Random object.
   * @param rating_map Concurrent rating map for gain computation.
   * @return Pair with: whether the node was moved to another cluster, whether
   * the previous cluster is now empty.
   */
  std::pair<bool, bool>
  handle_second_phase_node(const NodeID u, Random &rand, ConcurrentRatingMap &map) {
    if (derived_skip_node(u)) {
      return {false, false};
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = derived_cluster(u);

    const auto [best_cluster, gain] =
        find_best_cluster_second_phase(u, u_weight, u_cluster, rand, map);
    return try_node_move<true>(u, u_weight, u_cluster, best_cluster, gain);
  }

  struct ClusterSelectionState {
    Random &local_rand;
    NodeID u;
    NodeWeight u_weight;
    ClusterID initial_cluster;
    ClusterWeight initial_cluster_weight;
    ClusterID best_cluster;
    EdgeWeight best_gain;
    ClusterWeight best_cluster_weight;
    EdgeWeight overall_best_gain;
    ClusterID current_cluster;
    EdgeWeight current_gain;
    ClusterWeight current_cluster_weight;
  };

  struct LocalClusterSelectionState {
    EdgeWeight best_gain;
    ClusterID best_cluster;
    EdgeWeight favored_cluster_gain;
    ClusterID favored_cluster;
  };

  template <typename RatingMap>
  std::pair<ClusterID, EdgeWeight> find_best_cluster(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = derived_cluster_weight(u_cluster);
    ClusterSelectionState state{
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
      if (derived_accept_neighbor(u, v)) {
        const ClusterID v_cluster = derived_cluster(v);
        map[v_cluster] += w;

        if constexpr (Config::kUseLocalActiveSetStrategy) {
          is_interface_node |= v >= _num_active_nodes;
        }
      }
    };

    // As the compressed graph data structure has some overhead when imposing a limit on the number
    // of neighbors visited, we make a case distinction here, as the general case is not to restrict
    // the number of neighbors visited
    if (_max_num_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph->adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph->adjacent_nodes(u, _max_num_neighbors, add_to_rating_map);
    }

    if constexpr (Config::kUseActiveSetStrategy) {
      __atomic_store_n(&_active[u], 0, __ATOMIC_RELAXED);
    } else if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (!is_interface_node) {
        __atomic_store_n(&_active[u], 0, __ATOMIC_RELAXED);
      }
    }

    // After LP, we might want to use 2-hop clustering to merge nodes that could not find any
    // cluster to join for this, we store a favored cluster for each node u if:
    // (1) we actually use 2-hop clustering
    // (2) u is still in a singleton cluster (weight of node == weight of cluster)
    // (3) the cluster is light (at most half full)
    const bool store_favored_cluster =
        Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= derived_max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;
    ClusterID favored_cluster = derived_select_best_cluster(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );

    // If we couldn't join any cluster, we store the favored cluster
    if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
      _favored_clusters[u] = favored_cluster;
    }

    const EdgeWeight actual_gain = IFSTATS(state.best_gain - map[state.initial_cluster]);
    map.clear();
    return std::make_pair(state.best_cluster, actual_gain);
  }

  template <typename RatingMap>
  std::optional<std::pair<ClusterID, EdgeWeight>> find_best_cluster_first_phase(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = derived_cluster_weight(u_cluster);
    ClusterSelectionState state{
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
      if (derived_accept_neighbor(u, v)) {
        const ClusterID v_cluster = derived_cluster(v);
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

    // As the compressed graph data structure has some overhead when imposing a limit on the number
    // of neighbors visited, we make a case distinction here, as the general case is not to restrict
    // the number of neighbors visited
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
      __atomic_store_n(&_active[u], 0, __ATOMIC_RELAXED);
    } else if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (!is_interface_node) {
        __atomic_store_n(&_active[u], 0, __ATOMIC_RELAXED);
      }
    }

    // After LP, we might want to use 2-hop clustering to merge nodes that could not find any
    // cluster to join for this, we store a favored cluster for each node u if:
    // (1) we actually use 2-hop clustering
    // (2) u is still in a singleton cluster (weight of node == weight of cluster)
    // (3) the cluster is light (at most half full)
    const bool store_favored_cluster =
        Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= derived_max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;
    ClusterID favored_cluster = derived_select_best_cluster(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );

    // If we couldn't join any cluster, we store the favored cluster
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
    const ClusterWeight initial_cluster_weight = derived_cluster_weight(u_cluster);

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
        if (derived_accept_neighbor(u, v)) {
          const ClusterID v_cluster = derived_cluster(v);
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
      __atomic_store_n(&_active[u], 0, __ATOMIC_RELAXED);
    } else if constexpr (Config::kUseLocalActiveSetStrategy) {
      if (!is_interface_node) {
        __atomic_store_n(&_active[u], 0, __ATOMIC_RELAXED);
      }
    }

    // After LP, we might want to use 2-hop clustering to merge nodes that could not find any
    // cluster to join for this, we store a favored cluster for each node u if:
    // (1) we actually use 2-hop clustering
    // (2) u is still in a singleton cluster (weight of node == weight of cluster)
    // (3) the cluster is light (at most half full)
    const bool store_favored_cluster =
        Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= derived_max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;

    map.iterate_and_reset([&](const auto i, const auto &local_entries) {
      ClusterSelectionState local_state{
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

      const ClusterID local_favored_cluster = derived_select_best_cluster(
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

    const bool use_uniform_tie_breaking = _tie_breaking_strategy == TieBreakingStrategy::UNIFORM;
    if (use_uniform_tie_breaking) {
      auto &tie_breaking_clusters = _tie_breaking_clusters_ets.local();
      auto &tie_breaking_favored_clusters = _tie_breaking_favored_clusters_ets.local();

      EdgeWeight favored_cluster_gain = 0;
      for (LocalClusterSelectionState &local_state : _local_cluster_selection_states) {
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
      for (LocalClusterSelectionState &local_state : _local_cluster_selection_states) {
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

    // If we couldn't join any cluster, we store the favored cluster
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
    if (derived_cluster(u) != new_cluster) {
      const bool successful_weight_move = derived_move_cluster_weight(
          u_cluster, new_cluster, u_weight, derived_max_cluster_weight(new_cluster)
      );

      if (successful_weight_move) {
        derived_move_node(u, new_cluster);
        activate_neighbors<kParallelActivate>(u);
        IFSTATS(_expected_total_gain += gain);

        const bool decrement_cluster_count =
            Config::kTrackClusterCount && derived_cluster_weight(u_cluster) == 0;
        // do not update _current_num_clusters here to avoid fetch_add()
        return {true, decrement_cluster_count}; // did move, did reduce nonempty
                                                // cluster count?
      }
    }

    // did not move, did not reduce cluster count
    return {false, false};
  }

  /*!
   * Flags neighbors of a node that has been moved as active.
   *
   * @param u Node that was moved.
   */
  template <bool kParallel = false> void activate_neighbors(const NodeID u) {
    const auto activate_neighbors = [&](const NodeID v) {
      // call derived_activate_neighbor() even if we do not use the active set
      // strategy since the function might have side effects; the compiler
      // should remove it if it does not side effects
      if (derived_activate_neighbor(v)) {
        if constexpr (Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy) {
          __atomic_store_n(&_active[v], 1, __ATOMIC_RELAXED);
        }
      }
    };

    if constexpr (kParallel) {
      _graph->pfor_adjacent_nodes(
          u,
          std::numeric_limits<NodeID>::max(),
          20000,
          [&](const NodeID v, [[maybe_unused]] const EdgeWeight) { activate_neighbors(v); }
      );
    } else {
      _graph->adjacent_nodes(u, activate_neighbors);
    }
  }

  void match_isolated_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    handle_isolated_nodes_impl<true>(from, to);
  }

  void cluster_isolated_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    handle_isolated_nodes_impl<false>(from, to);
  }

  template <bool match>
  void handle_isolated_nodes_impl(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    constexpr ClusterID kInvalidClusterID = std::numeric_limits<ClusterID>::max();
    tbb::enumerable_thread_specific<ClusterID> current_cluster_ets(kInvalidClusterID);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, std::min(_graph->n(), to)),
        [&](tbb::blocked_range<NodeID> r) {
          ClusterID cluster = current_cluster_ets.local();

          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (_graph->degree(u) == 0) {
              const ClusterID cu = derived_cluster(u);

              if (cluster != kInvalidClusterID &&
                  derived_move_cluster_weight(
                      cu, cluster, derived_cluster_weight(cu), derived_max_cluster_weight(cluster)
                  )) {
                derived_move_node(u, cluster);
                if constexpr (match) {
                  cluster = kInvalidClusterID;
                }
              } else {
                cluster = cu;
              }
            }
          }

          current_cluster_ets.local() = cluster;
        }
    );
  }

  void match_two_hop_nodes_threadwise(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    handle_two_hop_nodes_threadwise_impl<true>(from, to);
  }

  void cluster_two_hop_nodes_threadwise(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    handle_two_hop_nodes_threadwise_impl<false>(from, to);
  }

  template <bool match>
  void handle_two_hop_nodes_threadwise_impl(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    static_assert(Config::kUseTwoHopClustering, "2-hop clustering is disabled");

    tbb::enumerable_thread_specific<DynamicFlatMap<ClusterID, NodeID>> matching_map_ets;

    auto is_considered_for_two_hop_clustering = [&](const NodeID u) {
      // Not considered: isolated node
      if (_graph->degree(u) == 0) {
        return false;
      }

      // If u did not join another cluster, there could still be other nodes that joined this
      // node's cluster: find out by checking the cluster weight
      auto check_cluster_weight = [&](const NodeID c_u) {
        const ClusterWeight current_weight = derived_cluster_weight(c_u);

        if (current_weight > derived_max_cluster_weight(c_u) / 2 ||
            current_weight != derived_initial_cluster_weight(c_u)) {
          // Not considered: not a singleton cluster; or its weight is too heavy
          return false;
        }

        return true;
      };

      // Not considered: joined another cluster
      if (_relabeled) {
        if (_moved[u]) {
          return false;
        }

        const ClusterID c_u = derived_cluster(u);
        return check_cluster_weight(c_u);
      } else {
        if (u != derived_cluster(u)) {
          return false;
        }

        // In this case c_u == u holds.
        return check_cluster_weight(u);
      }
    };

    auto handle_node = [&](DynamicFlatMap<ClusterID, NodeID> &matching_map, const NodeID u) {
      const ClusterID c_u = derived_cluster(u);
      ClusterID &rep_key = matching_map[_favored_clusters[u]];

      if (rep_key == 0) {
        rep_key = c_u + 1;
      } else {
        const ClusterID rep = rep_key - 1;

        const bool could_move_u_to_rep = derived_move_cluster_weight(
            c_u, rep, derived_cluster_weight(c_u), derived_max_cluster_weight(rep)
        );

        if constexpr (match) {
          KASSERT(could_move_u_to_rep);
          derived_move_node(u, rep);
          rep_key = 0;
        } else {
          if (could_move_u_to_rep) {
            derived_move_node(u, rep);
          } else {
            rep_key = c_u + 1;
          }
        }
      }
    };

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, std::min(to, _graph->n()), 512),
        [&](const tbb::blocked_range<NodeID> &r) {
          auto &matching_map = matching_map_ets.local();

          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (is_considered_for_two_hop_clustering(u)) {
              handle_node(matching_map, u);
            }
          }
        }
    );
  }

  void match_two_hop_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    handle_two_hop_nodes_impl<true>(from, to);
  }

  void cluster_two_hop_nodes(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    handle_two_hop_nodes_impl<false>(from, to);
  }

  template <bool match>
  void handle_two_hop_nodes_impl(
      const NodeID from = 0, const NodeID to = std::numeric_limits<ClusterID>::max()
  ) {
    static_assert(Config::kUseTwoHopClustering, "2-hop clustering is disabled");

    auto is_considered_for_two_hop_clustering = [&](const NodeID u) {
      // Skip nodes not considered for two-hop clustering
      if (_graph->degree(u) == 0) {
        // Not considered: isolated node
        return false;
      } else if (u != derived_cluster(u)) {
        // Not considered: joined another cluster
        return false;
      } else {
        // If u did not join another cluster, there could still be other nodes that joined this
        // node's cluster: find out by checking the cluster weight
        const ClusterWeight current_weight = derived_cluster_weight(u);
        if (current_weight > derived_max_cluster_weight(u) / 2 ||
            current_weight != derived_initial_cluster_weight(u)) {
          // Not considered: not a singleton cluster; or its weight is too heavy
          return false;
        }
      }

      return true;
    };

    // There could be edge cases where the favorite cluster of a node is itself a singleton cluster
    // (for instance, if a node joins another cluster during the first round, but moves out of the
    // cluster in the next round)
    // Since the following code is based on the ansumption that the favorite cluster of a node that
    // is considered for two-hop clustering it itself not considere for two-hop clustering, we fix
    // this situation by moving the nodes to their favorite cluster, if possible, here.
    tbb::parallel_for(from, std::min(to, _graph->n()), [&](const NodeID u) {
      if (is_considered_for_two_hop_clustering(u)) {
        const NodeID cluster = _favored_clusters[u];
        if (is_considered_for_two_hop_clustering(cluster) &&
            derived_move_cluster_weight(
                u, cluster, derived_cluster_weight(u), derived_max_cluster_weight(cluster)
            )) {
          derived_move_node(u, cluster);
          --_current_num_clusters;
        }
      } else {
        _favored_clusters[u] = u;
      }
    });

    KASSERT(
        [&] {
          for (NodeID u = from; u < std::min(to, _graph->n()); ++u) {
            if (_favored_clusters[u] >= _graph->n()) {
              LOG_WARNING << "favored cluster of node " << u
                          << " out of bounds: " << _favored_clusters[u] << " > " << _graph->n();
            }
            if (u != _favored_clusters[u] && is_considered_for_two_hop_clustering(u) &&
                is_considered_for_two_hop_clustering(_favored_clusters[u])) {
              LOG_WARNING << "node " << u << " (degree " << _graph->degree(u) << " )"
                          << " is considered for two-hop clustering, but its favored cluster "
                          << _favored_clusters[u] << " (degree "
                          << _graph->degree(_favored_clusters[u])
                          << ") is also considered for two-hop clustering";
              return false;
            }
          }
          return true;
        }(),
        "precondition for two-hop clustering violated: found favored clusters that could be joined",
        assert::heavy
    );

    // During label propagation, we store the best cluster for each node in _favored_cluster[]
    // regardless of whether there is enough space in the cluster for the node to join.
    // We now use this information to merge nodes that could not join any cluster, i.e.,
    // singleton-clusters by clustering or matching nodes that have favored cluster.

    tbb::parallel_for(from, std::min(to, _graph->n()), [&](const NodeID u) {
      if (should_stop()) {
        return;
      }

      // Skip nodes not considered for two-hop clustering
      if (!is_considered_for_two_hop_clustering(u)) {
        return;
      }

      // Invariant:
      // For each node u that is considered for two-hop clustering (i.e., nodes for which the
      // following lines of code are executed), _favored_clusters[u] refers to node which *IS NOT*
      // considered for two-hop matching.
      //
      // Reasoning:
      // KASSERT()
      //
      // Conclusion:
      // We can use _favored_clusters[u] to build the two-hop clusters.

      const NodeID C = __atomic_load_n(&_favored_clusters[u], __ATOMIC_RELAXED);
      auto &sync = _favored_clusters[C];

      do {
        NodeID cluster = sync;

        if (cluster == C) {
          if (__atomic_compare_exchange_n(
                  &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
              )) {
            // We are done: other nodes will join our cluster
            break;
          }
          if (cluster == C) {
            continue;
          }
        }

        // Invariant: cluster is a node with favored cluster C
        KASSERT(__atomic_load_n(&_favored_clusters[cluster], __ATOMIC_RELAXED) == C);

        // Try to join the cluster:
        if constexpr (match) {
          // Matching mode: try to build a cluster only containing nodes "cluster" and "u"
          if (__atomic_compare_exchange_n(
                  &sync, &cluster, C, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
              )) {
            [[maybe_unused]] const bool success = derived_move_cluster_weight(
                u, cluster, derived_cluster_weight(u), derived_max_cluster_weight(cluster)
            );
            KASSERT(
                success,
                "node " << u << " could be matched with node " << cluster << ": "
                        << derived_cluster_weight(u) << " + " << derived_cluster_weight(cluster)
                        << " > " << derived_max_cluster_weight(cluster)
            );

            derived_move_node(u, cluster);
            --_current_num_clusters;

            // We are done: build a cluster with "cluster", reset "sync" to C
            break;
          }
        } else {
          // Clustering mode: try to join cluster "cluster" if the weight constraint permits it,
          // otherwise try to start a new cluster
          if (derived_move_cluster_weight(
                  u, cluster, derived_cluster_weight(u), derived_max_cluster_weight(cluster)
              )) {
            derived_move_node(u, cluster);
            --_current_num_clusters;

            // We are done: joined cluster "cluster"
            break;
          } else if (__atomic_compare_exchange_n(
                         &sync, &cluster, C, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
                     )) {
            // We are done: start a new cluster
            break;
          }
        }
      } while (true);
    });
  }

private:
  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
            if constexpr (Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy) {
              _active[u] = 1;
            }

            const ClusterID initial_cluster = derived_initial_cluster(u);
            derived_init_cluster(u, initial_cluster);
            if constexpr (Config::kUseTwoHopClustering) {
              _favored_clusters[u] = initial_cluster;
            }

            derived_reset_node_state(u);
          });
        },
        [&] {
          tbb::parallel_for<ClusterID>(0, _initial_num_clusters, [&](const ClusterID cluster) {
            derived_init_cluster_weight(cluster, derived_initial_cluster_weight(cluster));
          });
        }
    );
    IFSTATS(_expected_total_gain = 0);
    _current_num_clusters = _initial_num_clusters;
    _relabeled = false;
  }

private: // CRTP calls
  //! Return current cluster ID of  node \c u.
  [[nodiscard]] ClusterID derived_cluster(const NodeID u) {
    return static_cast<Derived *>(this)->cluster(u);
  }

  //! Initially place \c u in cluster \cluster.
  void derived_init_cluster(const NodeID u, const ClusterID cluster) {
    static_cast<Derived *>(this)->init_cluster(u, cluster);
  }

  //! Change cluster of node \c u to \c cluster.
  void derived_move_node(const NodeID u, const ClusterID cluster) {
    static_cast<Derived *>(this)->move_node(u, cluster);
  }

  //! Return current weight of cluster \c cluster.
  [[nodiscard]] ClusterWeight derived_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->cluster_weight(cluster);
  }

  //! Initially set weight of cluster \cluster to \c weight.
  void derived_init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    static_cast<Derived *>(this)->init_cluster_weight(cluster, weight);
  }

  //! Attempt to move \c delta weight from cluster \c old_cluster to \c
  //! new_cluster, which can take at most \c max_weight weight.
  [[nodiscard]] bool derived_move_cluster_weight(
      const ClusterID old_cluster,
      const ClusterID new_cluster,
      const ClusterWeight delta,
      const ClusterWeight max_weight
  ) {
    return static_cast<Derived *>(this)->move_cluster_weight(
        old_cluster, new_cluster, delta, max_weight
    );
  }

  //! Reassigns the cluster weights, which is required for relabeling the clusters.
  void derived_reassign_cluster_weights(
      const StaticArray<ClusterID> &mapping, const ClusterID num_new_clusters
  ) {
    static_cast<Derived *>(this)->reassign_cluster_weights(mapping, num_new_clusters);
  }

  //! Return the maximum weight of cluster \c cluster.
  [[nodiscard]] ClusterWeight derived_max_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->max_cluster_weight(cluster);
  }

  template <typename RatingMap>
  [[nodiscard]] ClusterID derived_select_best_cluster(
      const bool store_favored_cluster,
      const EdgeWeight gain_delta,
      ClusterSelectionState &state,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    return static_cast<Derived *>(this)->select_best_cluster(
        store_favored_cluster,
        gain_delta,
        state,
        map,
        tie_breaking_clusters,
        tie_breaking_favored_clusters
    );
  }

  void derived_reset_node_state(const NodeID u) {
    static_cast<Derived *>(this)->reset_node_state(u);
  }

  [[nodiscard]] inline bool derived_accept_neighbor(const NodeID u, const NodeID v) {
    return static_cast<Derived *>(this)->accept_neighbor(u, v);
  }

  [[nodiscard]] inline bool derived_activate_neighbor(const NodeID u) {
    return static_cast<Derived *>(this)->activate_neighbor(u);
  }

  [[nodiscard]] ClusterID derived_initial_cluster(const NodeID u) {
    return static_cast<Derived *>(this)->initial_cluster(u);
  }

  [[nodiscard]] ClusterWeight derived_initial_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->initial_cluster_weight(cluster);
  }

  [[nodiscard]] bool derived_skip_node(const NodeID node) {
    return static_cast<Derived *>(this)->skip_node(node);
  }

protected: // Default implementations
  void reset_node_state(const NodeID /* node */) {}

  [[nodiscard]] inline bool accept_neighbor(const NodeID /* u */, const NodeID /* v */) {
    return true;
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID /* node */) {
    return true;
  }

  [[nodiscard]] inline ClusterID initial_cluster(const NodeID u) {
    return derived_cluster(u);
  }

  [[nodiscard]] inline ClusterWeight initial_cluster_weight(const ClusterID cluster) {
    return derived_cluster_weight(cluster);
  }

  [[nodiscard]] inline bool skip_node(const NodeID /* node */) {
    return false;
  }

protected: // Members
  //! Graph we operate on, or \c nullptr if \c initialize has not been called yet.
  const Graph *_graph = nullptr;

  //! The number of non-empty clusters before we ran the first iteration of
  //! label propagation.
  ClusterID _initial_num_clusters;

  //! The current number of non-empty clusters. Only meaningful if empty
  //! clusters are being counted.
  parallel::Atomic<ClusterID> _current_num_clusters;

  //! We stop label propagation if the number of non-empty clusters falls below
  //! this threshold. Only has an effect if empty clusters are being counted.
  ClusterID _desired_num_clusters = 0;

  //! We do not move nodes with a degree higher than this. However, other nodes
  //! may still be moved to the cluster of with degree larger than this
  //! threshold.
  NodeID _max_degree = std::numeric_limits<NodeID>::max();

  //! When computing the gain values for a node, this is an upper limit on the
  //! number of neighbors of the nodes we consider. Any more neighbors are
  //! ignored.
  NodeID _max_num_neighbors = std::numeric_limits<NodeID>::max();

  //! The label propagation implementation that is used.
  LabelPropagationImplementation _impl;

  //! The tie breaking strategy that is used.
  TieBreakingStrategy _tie_breaking_strategy;

  //! Whether to relabel the clusters before the second phase.
  bool _relabel_before_second_phase;

  //! Thread-local map to compute gain values (static).
  tbb::enumerable_thread_specific<RatingMap> _rating_map_ets;

  //! Thread-local map to compute gain values (growing).
  tbb::enumerable_thread_specific<GrowingRatingMap> _growing_rating_map_ets;

  //! Thread-local vector to hold clusters considered for uniform tie-breaking.
  tbb::enumerable_thread_specific<ScalableVector<ClusterID>> _tie_breaking_clusters_ets;

  //! Thread-local vector to hold favored clusters considered for uniform tie-breaking.
  tbb::enumerable_thread_specific<ScalableVector<ClusterID>> _tie_breaking_favored_clusters_ets;

  //! Vector of local cluster selection states where each entry is owned by a parallel task.
  CacheAlignedVector<LocalClusterSelectionState> _local_cluster_selection_states;

  //! Flags nodes with at least one node in its neighborhood that changed
  //! clusters during the last iteration. Nodes without this flag set must not
  //! be considered in the next iteration.
  StaticArray<std::uint8_t> _active;

  //! Flags nodes that joined another cluster. This information is used during 2-hop clustering when
  //! we relabel the clusters.
  StaticArray<std::uint8_t> _moved;

  //! Store whether we relabeled the clusters and thus have to use the information of the _moved
  //! array for 2-hop clustering.
  bool _relabeled;

  //! If a node cannot join any cluster during an iteration, this vector stores
  //! the node's highest rated cluster independent of the maximum cluster
  //! weight. This information is used during 2-hop clustering.
  StaticArray<ClusterID> _favored_clusters;

  //! The nodes which should be processed in the second phase.
  tbb::concurrent_vector<NodeID> _second_phase_nodes;

  //! If statistics are enabled, this is the sum of the gain of all moves that
  //! were performed. If executed single-thread, this should be equal to the
  //! reduction of the edge cut.
  parallel::Atomic<EdgeWeight> _expected_total_gain;

private:
  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
  ClusterID _prev_num_clusters = 0;
};

/*!
 * Parallel label propagation template that iterates over nodes in their natural
 * order.
 * @tparam Derived Derived subclass for static polymorphism.
 * @tparam Config Algorithmic configuration and data types.
 */
template <typename Derived, typename Config, typename Graph>
class InOrderLabelPropagation : public LabelPropagation<Derived, Config, Graph> {
  static_assert(std::is_base_of_v<LabelPropagationConfig, Config>);
  SET_DEBUG(false);

  using Base = LabelPropagation<Derived, Config, Graph>;

protected:
  using NodeID = typename Base::NodeID;
  using NodeWeight = typename Base::NodeWeight;
  using EdgeID = typename Base::EdgeID;
  using EdgeWeight = typename Base::EdgeWeight;
  using ClusterID = typename Base::ClusterID;
  using ClusterWeight = typename Base::ClusterWeight;

  using Base::handle_node;
  using Base::set_max_degree;
  using Base::set_max_num_neighbors;
  using Base::should_stop;

  NodeID
  perform_iteration(const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()) {
    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, std::min(_graph->n(), to)),
        [&](const auto &r) {
          EdgeID work_since_update = 0;
          NodeID num_removed_clusters = 0;

          auto &num_moved_nodes = num_moved_nodes_ets.local();
          auto &rand = Random::instance();
          auto &rating_map = _rating_map_ets.local();

          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (_graph->degree(u) > _max_degree) {
              continue;
            }

            if constexpr (Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy) {
              if (!__atomic_load_n(&_active[u], __ATOMIC_RELAXED)) {
                continue;
              }
            }

            if (work_since_update > Config::kMinChunkSize) {
              if (Base::should_stop()) {
                return;
              }

              _current_num_clusters -= num_removed_clusters;
              work_since_update = 0;
              num_removed_clusters = 0;
            }

            const auto [moved_node, emptied_cluster] = handle_node(u, rand, rating_map);
            work_since_update += _graph->degree(u);
            if (moved_node) {
              ++num_moved_nodes;
            }
            if (emptied_cluster) {
              ++num_removed_clusters;
            }
          }
        }
    );

    return num_moved_nodes_ets.combine(std::plus{});
  }

  using Base::_active;
  using Base::_current_num_clusters;
  using Base::_graph;
  using Base::_max_degree;
  using Base::_rating_map_ets;
};

template <typename NodeID> struct AbstractChunk {
  NodeID start;
  NodeID end;
};

struct Bucket {
  std::size_t start;
  std::size_t end;
};

/*!
 * Parallel label propagation template that iterates over nodes in chunk random
 * order.
 * @tparam Derived Derived subclass for static polymorphism.
 * @tparam Config Algorithmic configuration and data types.
 */
template <typename Derived, typename Config, typename Graph>
class ChunkRandomLabelPropagation : public LabelPropagation<Derived, Config, Graph> {
  static_assert(std::is_base_of_v<LabelPropagationConfig, Config>);
  SET_DEBUG(false);

  using Base = LabelPropagation<Derived, Config, Graph>;

protected:
  using NodeID = typename Base::NodeID;
  using NodeWeight = typename Base::NodeWeight;
  using EdgeID = typename Base::EdgeID;
  using EdgeWeight = typename Base::EdgeWeight;

  using ClusterID = typename Base::ClusterID;
  using ClusterWeight = typename Base::ClusterWeight;

  using RatingMap = typename Base::RatingMap;
  using ConcurrentRatingMap = typename Base::ConcurrentRatingMap;
  using GrowingRatingMap = typename Base::GrowingRatingMap;

  using LabelPropagationImplementation = Base::LabelPropagationImplementation;

  using Base::handle_first_phase_node;
  using Base::handle_node;
  using Base::handle_second_phase_node;
  using Base::relabel_clusters;
  using Base::set_max_degree;
  using Base::set_max_num_neighbors;
  using Base::should_stop;

  using Permutations =
      RandomPermutations<NodeID, Config::kPermutationSize, Config::kNumberOfNodePermutations>;
  using Chunk = AbstractChunk<NodeID>;

public:
  //! The data strucutres that are stored on the heap and used by label propagation.
  using DataStructures = std::tuple<
      tbb::enumerable_thread_specific<RatingMap>,
      tbb::enumerable_thread_specific<GrowingRatingMap>,
      ConcurrentRatingMap,
      tbb::enumerable_thread_specific<ScalableVector<ClusterID>>,
      StaticArray<uint8_t>,
      StaticArray<uint8_t>,
      StaticArray<ClusterID>,
      tbb::concurrent_vector<NodeID>,
      tbb::enumerable_thread_specific<std::vector<NodeID>>,
      tbb::enumerable_thread_specific<std::size_t>,
      tbb::enumerable_thread_specific<std::vector<Chunk>>,
      std::vector<Chunk>,
      std::vector<Bucket>>;

  /*!
   * Sets the data structures to use, which can save memory space when (unused) data structures are
   * already in memory.
   *
   * @param structs The data structures to use.
   */
  void setup(DataStructures structs) {
    auto
        [rating_map_ets,
         growing_rating_map_ets,
         concurrent_rating_map,
         tie_breaking_clusters_ets,
         active,
         moved,
         favored_clusters,
         second_phase_nodes,
         sub_chunk_permutation_ets,
         num_chunks_ets,
         chunks_ets,
         chunks,
         buckets] = std::move(structs);
    _rating_map_ets = std::move(rating_map_ets);
    _growing_rating_map_ets = std::move(growing_rating_map_ets);
    _concurrent_rating_map = std::move(concurrent_rating_map);
    _tie_breaking_clusters_ets = std::move(tie_breaking_clusters_ets);
    _active = std::move(active);
    _moved = std::move(moved);
    _favored_clusters = std::move(favored_clusters);
    _second_phase_nodes = std::move(second_phase_nodes);
    _sub_chunk_permutation_ets = std::move(sub_chunk_permutation_ets);
    _num_chunks_ets = std::move(num_chunks_ets);
    _chunks_ets = std::move(chunks_ets);
    _chunks = std::move(chunks);
    _buckets = std::move(buckets);
  }

  /*!
   * Returns ownership of the data structures that are stored on the heap.
   *
   * @return Ownership of the data structures that are stored on the heap.
   */
  DataStructures release() {
    return std::make_tuple(
        std::move(_rating_map_ets),
        std::move(_growing_rating_map_ets),
        std::move(_concurrent_rating_map),
        std::move(_tie_breaking_clusters_ets),
        std::move(_active),
        std::move(_moved),
        std::move(_favored_clusters),
        std::move(_second_phase_nodes),
        std::move(_sub_chunk_permutation_ets),
        std::move(_num_chunks_ets),
        std::move(_chunks_ets),
        std::move(_chunks),
        std::move(_buckets)
    );
  }

protected:
  ChunkRandomLabelPropagation(Permutations &permutations) : _random_permutations(permutations) {}

  void free() {
    Base::free();

    _chunks.clear();
    _chunks.shrink_to_fit();

    _buckets.clear();
    _buckets.shrink_to_fit();

    _concurrent_rating_map.free();
  }

  void initialize(const Graph *graph, const ClusterID num_clusters) {
    Base::initialize(graph, num_clusters);
    _chunks.clear();
    _buckets.clear();
  }

  /**
   * Performs label propagation on local nodes in range [from, to) in
   * chunk-randomized order.
   *
   * The randomization works in multiple steps:
   * - Nodes within the iteration order are split into chunks of consecutive
   * nodes. The size of each chunk is determined by
   * LegacyLabelPropagationConfig::kMinChunkSize, which is a lower bound on the sum of
   * the degrees assigned to a chunk (nodes are assigned to a chunk until the
   * limit is exceeded).
   * - Afterwards, the order of chunk is shuffled.
   * - Finally, chunks are processed in parallel. To this end, the nodes
   * assigned to a chunk are once more split into sub-chunks, which are then
   * processed sequentially and in-order; however, within a sub-chunk, nodes are
   * once more shuffled.
   * - If available, degree buckets are respected: chunks of smaller buckets are
   * processed before chunks of larger buckets.
   *
   * @param from First node in the iteration range.
   * @param to First node that is not part of the iteration range.
   * @return Number of nodes that where moved to new blocks / clusters.
   */
  NodeID
  perform_iteration(const NodeID from = 0, const NodeID to = std::numeric_limits<NodeID>::max()) {
    if (from != 0 || to != std::numeric_limits<NodeID>::max()) {
      _chunks.clear();
    }
    if (_chunks.empty()) {
      init_chunks(from, to);
    }
    shuffle_chunks();

    switch (_impl) {
    case LabelPropagationImplementation::GROWING_HASH_TABLES:
      perform(_growing_rating_map_ets);
      break;
    case LabelPropagationImplementation::SINGLE_PHASE:
      perform(_rating_map_ets);
      break;
    case LabelPropagationImplementation::TWO_PHASE:
      const NodeID initial_num_clusters = _initial_num_clusters;
      const auto [num_processed_nodes, num_moved_nodes_first_phase] = perform_first_phase();

      const NodeID num_second_phase_nodes = _second_phase_nodes.size();
      if (num_second_phase_nodes > 0) {
        if (_relabel_before_second_phase) {
          relabel_clusters();
        }

        perform_second_phase();
      }

      const NodeID num_moved_nodes = _num_moved_nodes_ets.combine(std::plus{});
      if constexpr (kDebug) {
        LOG << "Label Propagation";
        LOG << " Initial clusters: " << initial_num_clusters << " clusters";
        LOG << " First Phase:";
        LOG << "  Processed: " << (num_processed_nodes - num_second_phase_nodes) << " nodes";
        LOG << "  Moved: " << num_moved_nodes_first_phase << " nodes";
        if (_relabel_before_second_phase) {
          LOG << " Clusters after relabeling: " << _initial_num_clusters << " clusters";
        }
        LOG << " Second Phase:";
        LOG << "  Processed: " << num_second_phase_nodes << " nodes";
        LOG << "  Moved: " << (num_moved_nodes - num_moved_nodes_first_phase) << " nodes";
        LOG;
      }
      break;
    }

    const NodeID num_moved_nodes = _num_moved_nodes_ets.combine(std::plus{});
    _num_processed_nodes_ets.clear();
    _num_moved_nodes_ets.clear();
    return num_moved_nodes;
  }

private:
  void init_chunks(const NodeID from, NodeID to) {
    _chunks.clear();
    _buckets.clear();

    to = std::min(to, _graph->n());

    const auto max_bucket =
        std::min<std::size_t>(math::floor_log2(_max_degree), _graph->number_of_buckets());
    const EdgeID max_chunk_size = std::max<EdgeID>(Config::kMinChunkSize, std::sqrt(_graph->m()));
    const NodeID max_node_chunk_size =
        std::max<NodeID>(Config::kMinChunkSize, std::sqrt(_graph->n()));

    NodeID position = 0;
    for (std::size_t bucket = 0; bucket < max_bucket; ++bucket) {
      if (position + _graph->bucket_size(bucket) < from || _graph->bucket_size(bucket) == 0) {
        position += _graph->bucket_size(bucket);
        continue;
      }
      if (position >= to) {
        break;
      }

      NodeID remaining_bucket_size = _graph->bucket_size(bucket);
      if (from > _graph->first_node_in_bucket(bucket)) {
        remaining_bucket_size -= from - _graph->first_node_in_bucket(bucket);
      }
      const std::size_t bucket_size =
          std::min<NodeID>({remaining_bucket_size, to - position, to - from});

      parallel::Atomic<NodeID> offset = 0;

      const std::size_t bucket_start = std::max(_graph->first_node_in_bucket(bucket), from);

      tbb::parallel_for(
          static_cast<int>(0),
          tbb::this_task_arena::max_concurrency(),
          [&](const int) {
            auto &chunks = _chunks_ets.local();
            auto &num_chunks = _num_chunks_ets.local();

            while (offset < bucket_size) {
              const NodeID begin = offset.fetch_add(max_node_chunk_size);
              if (begin >= bucket_size) {
                break;
              }
              const NodeID end = std::min<NodeID>(begin + max_node_chunk_size, bucket_size);

              EdgeID current_chunk_size = 0;
              NodeID chunk_start = bucket_start + begin;

              for (NodeID i = begin; i < end; ++i) {
                const NodeID u = bucket_start + i;
                current_chunk_size += _graph->degree(u);
                if (current_chunk_size >= max_chunk_size) {
                  chunks.push_back({chunk_start, u + 1});
                  chunk_start = u + 1;
                  current_chunk_size = 0;
                  ++num_chunks;
                }
              }

              if (current_chunk_size > 0) {
                chunks.push_back(
                    {static_cast<NodeID>(chunk_start), static_cast<NodeID>(bucket_start + end)}
                );
                ++num_chunks;
              }
            }
          }
      );

      std::size_t num_chunks = 0;
      for (auto &local_num_chunks : _num_chunks_ets) {
        num_chunks += local_num_chunks;
        local_num_chunks = 0;
      }

      const std::size_t chunks_start = _chunks.size();
      parallel::Atomic<std::size_t> pos = chunks_start;
      _chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(_chunks_ets.range(), [&](auto &r) {
        for (auto &chunk : r) {
          const std::size_t local_pos = pos.fetch_add(chunk.size());
          std::copy(chunk.begin(), chunk.end(), _chunks.begin() + local_pos);
          chunk.clear();
        }
      });

      _buckets.push_back({chunks_start, _chunks.size()});

      position += _graph->bucket_size(bucket);
    }

    // Make sure that we cover all nodes in [from, to)
    KASSERT(
        [&] {
          std::vector<bool> hit(to - from);
          for (const auto &[start, end] : _chunks) {
            KASSERT(start <= end, "");
            EdgeWeight total_work = 0;

            for (NodeID u = start; u < end; ++u) {
              KASSERT(from <= u, "");
              KASSERT(u < to, "");
              KASSERT(!hit[u - from], "");

              hit[u - from] = true;
              total_work += _graph->degree(u);
            }
          }

          for (NodeID u = 0; u < to - from; ++u) {
            KASSERT(_graph->degree(u) == 0u || hit[u]);
          }

          return true;
        }(),
        "",
        assert::heavy
    );
  }

  void shuffle_chunks() {
    tbb::parallel_for<std::size_t>(0, _buckets.size(), [&](const std::size_t i) {
      const auto &bucket = _buckets[i];
      Random::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

  template <typename RatingMapETS> void perform(RatingMapETS &rating_map_ets) {
    parallel::Atomic<std::size_t> next_chunk = 0;
    DBG << "Number of chunks: " << _chunks.size();

    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](std::size_t) {
      if (should_stop()) {
        return;
      }

      auto &local_num_processed_nodes = _num_processed_nodes_ets.local();
      auto &local_num_moved_nodes = _num_moved_nodes_ets.local();
      auto &local_rand = Random::instance();
      auto &local_rating_map = rating_map_ets.local();
      auto &tie_breaking_clusters = _tie_breaking_clusters_ets.local();
      auto &tie_breaking_favored_clusters = _tie_breaking_favored_clusters_ets.local();
      NodeID num_removed_clusters = 0;

      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = _chunks[chunk_id];
      const auto &permutation = _random_permutations.get(local_rand);

      DBG << chunk_id << " of " << _chunks.size() << " for " << sched_getcpu();

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / Config::kPermutationSize);

      auto &sub_chunk_permutation = _sub_chunk_permutation_ets.local();
      if (sub_chunk_permutation.size() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      local_rand.shuffle(
          sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks
      );

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < Config::kPermutationSize; ++i) {
          const NodeID u = chunk.start +
                           Config::kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % Config::kPermutationSize];
          if (u >= chunk.end) {
            continue;
          }

          if constexpr (Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy) {
            if (!__atomic_load_n(&_active[u], __ATOMIC_RELAXED)) {
              continue;
            }
          }

          const NodeID degree = _graph->degree(u);
          if (degree < _max_degree) {
            const auto [moved_node, emptied_cluster] = handle_node(
                u,
                local_rand,
                local_rating_map,
                tie_breaking_clusters,
                tie_breaking_favored_clusters
            );

            ++local_num_processed_nodes;
            if (moved_node) {
              ++local_num_moved_nodes;
            }
            if (emptied_cluster) {
              ++num_removed_clusters;
            }
          }
        }
      }

      _current_num_clusters -= num_removed_clusters;
    });
  }

  std::pair<std::size_t, std::size_t> perform_first_phase() {
    SCOPED_HEAP_PROFILER("First phase");
    SCOPED_TIMER("First phase");

    parallel::Atomic<std::size_t> next_chunk = 0;
    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](const std::size_t) {
      if (should_stop()) {
        return;
      }

      auto &local_num_processed_nodes = _num_processed_nodes_ets.local();
      auto &local_num_moved_nodes = _num_moved_nodes_ets.local();
      auto &local_rand = Random::instance();
      auto &local_rating_map = _rating_map_ets.local();
      auto &tie_breaking_clusters = _tie_breaking_clusters_ets.local();
      auto &tie_breaking_favored_clusters = _tie_breaking_favored_clusters_ets.local();
      NodeID num_removed_clusters = 0;

      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = _chunks[chunk_id];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / Config::kPermutationSize);

      auto &sub_chunk_permutation = _sub_chunk_permutation_ets.local();
      if (sub_chunk_permutation.size() < num_sub_chunks) {
        sub_chunk_permutation.resize(num_sub_chunks);
      }

      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks, 0);
      local_rand.shuffle(
          sub_chunk_permutation.begin(), sub_chunk_permutation.begin() + num_sub_chunks
      );

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < Config::kPermutationSize; ++i) {
          const NodeID u = chunk.start +
                           Config::kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % Config::kPermutationSize];
          if (u >= chunk.end) {
            continue;
          }

          if constexpr (Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy) {
            if (!__atomic_load_n(&_active[u], __ATOMIC_RELAXED)) {
              continue;
            }
          }

          const NodeID degree = _graph->degree(u);
          if (degree < _max_degree) {
            ++local_num_processed_nodes;

            const auto [moved_node, emptied_cluster] = handle_first_phase_node(
                u,
                local_rand,
                local_rating_map,
                tie_breaking_clusters,
                tie_breaking_favored_clusters
            );
            if (moved_node) {
              ++local_num_moved_nodes;

              if (_relabeled) {
                _moved[u] = 1;
              }
            }
            if (emptied_cluster) {
              ++num_removed_clusters;
            }
          }
        }
      }

      _current_num_clusters -= num_removed_clusters;
    });

    return std::make_pair(
        _num_processed_nodes_ets.combine(std::plus{}), _num_moved_nodes_ets.combine(std::plus{})
    );
  }

  void perform_second_phase() {
    SCOPED_HEAP_PROFILER("Second phase");
    SCOPED_TIMER("Second phase");

    const std::size_t num_clusters = _initial_num_clusters;
    if (_concurrent_rating_map.capacity() < num_clusters) {
      _concurrent_rating_map.resize(num_clusters);
    }

    auto &num_moved_nodes = _num_moved_nodes_ets.local();
    auto &rand = Random::instance();
    for (const NodeID u : _second_phase_nodes) {
      const auto [moved_node, emptied_cluster] =
          handle_second_phase_node(u, rand, _concurrent_rating_map);

      if (moved_node) {
        ++num_moved_nodes;

        if (_relabeled) {
          _moved[u] = 1;
        }
      }

      if (emptied_cluster) {
        --_current_num_clusters;
      }
    }

    _second_phase_nodes.clear();
  }

protected:
  using Base::_active;
  using Base::_current_num_clusters;
  using Base::_favored_clusters;
  using Base::_graph;
  using Base::_growing_rating_map_ets;
  using Base::_impl;
  using Base::_initial_num_clusters;
  using Base::_max_degree;
  using Base::_moved;
  using Base::_rating_map_ets;
  using Base::_relabel_before_second_phase;
  using Base::_relabeled;
  using Base::_second_phase_nodes;
  using Base::_tie_breaking_clusters_ets;
  using Base::_tie_breaking_favored_clusters_ets;

  Permutations &_random_permutations;
  tbb::enumerable_thread_specific<std::vector<NodeID>> _sub_chunk_permutation_ets;
  tbb::enumerable_thread_specific<std::size_t> _num_chunks_ets;
  tbb::enumerable_thread_specific<std::vector<Chunk>> _chunks_ets;
  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;

  tbb::enumerable_thread_specific<NodeID> _num_processed_nodes_ets;
  tbb::enumerable_thread_specific<NodeID> _num_moved_nodes_ets;

  ConcurrentRatingMap _concurrent_rating_map;
};

template <typename NodeID, typename ClusterID> class NonatomicClusterVectorRef {
public:
  void init_clusters_ref(StaticArray<NodeID> &clustering) {
    _clusters = &clustering;
  }

  void init_cluster(const NodeID node, const ClusterID cluster) {
    move_node(node, cluster);
  }

  [[nodiscard]] ClusterID cluster(const NodeID node) {
    KASSERT(node < _clusters->size());
    return __atomic_load_n(&_clusters->at(node), __ATOMIC_RELAXED);
  }

  void move_node(const NodeID node, const ClusterID cluster) {
    KASSERT(node < _clusters->size());
    __atomic_store_n(&_clusters->at(node), cluster, __ATOMIC_RELAXED);
  }

private:
  StaticArray<ClusterID> *_clusters = nullptr;
};

template <typename ClusterID, typename ClusterWeight> class OwnedRelaxedClusterWeightVector {
public:
  using ClusterWeights = StaticArray<ClusterWeight>;

  void allocate_cluster_weights(const ClusterID num_clusters) {
    if (_cluster_weights.size() < num_clusters) {
      _cluster_weights.resize(num_clusters);
    }
  }

  void free() {
    _cluster_weights.free();
  }

  void setup_cluster_weights(ClusterWeights cluster_weights) {
    _cluster_weights = std::move(cluster_weights);
  }

  ClusterWeights take_cluster_weights() {
    return std::move(_cluster_weights);
  }

  void reset_cluster_weights() {}

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights[cluster] = weight;
  }

  ClusterWeight cluster_weight(const ClusterID cluster) {
    return __atomic_load_n(&_cluster_weights[cluster], __ATOMIC_RELAXED);
  }

  bool move_cluster_weight(
      const ClusterID old_cluster,
      const ClusterID new_cluster,
      const ClusterWeight delta,
      const ClusterWeight max_weight
  ) {
    if (_cluster_weights[new_cluster] + delta <= max_weight) {
      __atomic_fetch_add(&_cluster_weights[new_cluster], delta, __ATOMIC_RELAXED);
      __atomic_fetch_sub(&_cluster_weights[old_cluster], delta, __ATOMIC_RELAXED);
      return true;
    }

    return false;
  }

  void reassign_cluster_weights(
      const StaticArray<ClusterID> &mapping, const ClusterID num_new_clusters
  ) {
    RECORD("new_cluster_weights") ClusterWeights new_cluster_weights(num_new_clusters);

    tbb::parallel_for(
        tbb::blocked_range<ClusterID>(0, _cluster_weights.size()),
        [&](const auto &r) {
          for (ClusterID u = r.begin(); u != r.end(); ++u) {
            ClusterWeight weight = _cluster_weights[u];

            if (weight != 0) {
              ClusterID new_cluster_id = mapping[u] - 1;
              new_cluster_weights[new_cluster_id] = weight;
            }
          }
        }
    );

    _cluster_weights = std::move(new_cluster_weights);
  }

private:
  ClusterWeights _cluster_weights;
};

} // namespace kaminpar
