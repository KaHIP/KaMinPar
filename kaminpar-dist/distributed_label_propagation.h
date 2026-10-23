/*******************************************************************************
 * Distributed label propagation using shared node-processing components.
 *
 * @file:   distributed_label_propagation.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <atomic>
#include <type_traits>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/logger.h"

#include "kaminpar-shm/algorithms/iteration_order.h"
#include "kaminpar-shm/algorithms/label_propagation/active_set.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/node_processing.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {

struct LabelPropagationConfig {
  // Data structure used to accumulate edge weights for gain value calculation
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID>;

  // Data type for cluster IDs and weights
  using ClusterID = void;
  using ClusterWeight = void;

  static constexpr bool kUseActualGain = false;

  // Opt in only when strictly smaller gains are rejected without RNG or other side effects.
  static constexpr bool kSkipLosingClusterWeights = false;

  static constexpr bool kUseActiveSetStrategy = true;
};

template <typename RatingMap> struct LabelPropagationMemoryContext {
  tbb::enumerable_thread_specific<RatingMap> rating_map_ets;
  shm::lp::ActiveSet active;
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
  using RatingMap = typename Config::RatingMap;
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using ClusterID = typename Config::ClusterID;
  using ClusterWeight = typename Config::ClusterWeight;

public:
  void enable_active_set() {
    if constexpr (!Config::kUseActiveSetStrategy) {
      LOG_WARNING << "Active set enabled, but feature is not included in this build";
    }
    _use_active_set = true;
  }

  void enable_local_active_set() {
    if constexpr (!Config::kUseActiveSetStrategy) {
      LOG_WARNING << "Active set enabled, but feature is not included in this build";
    }
    _use_local_active_set = true;
  }

  void set_max_degree(const NodeID max_degree) {
    _max_degree = max_degree;
  }
  void set_max_num_neighbors(const ClusterID max_num_neighbors) {
    _max_num_neighbors = max_num_neighbors;
  }
  [[nodiscard]] EdgeWeight expected_total_gain() const {
    return _expected_total_gain;
  }

  void setup(LabelPropagationMemoryContext<RatingMap> &memory_context) {
    _rating_map_ets = std::move(memory_context.rating_map_ets);
    _active = std::move(memory_context.active);
  }

  LabelPropagationMemoryContext<RatingMap> release() {
    return {
        std::move(_rating_map_ets),
        std::move(_active),
    };
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
    if constexpr (Config::kUseActiveSetStrategy) {
      if (_use_active_set || _use_local_active_set) {
        _active.resize(_use_local_active_set ? _num_nodes : _num_active_nodes);
      }
    }

    if (_rating_map_ets.empty()) {
      _rating_map_ets =
          tbb::enumerable_thread_specific<RatingMap>([&_num_clusters = _num_clusters] {
            return RatingMap(_num_clusters);
          });
    } else if (_prev_num_clusters < _num_clusters) {
      for (auto &rating_map : _rating_map_ets) {
        rating_map.change_max_size(_num_clusters);
      }
    }
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
    reset_state();
    _iteration_order_initialized = false;
  }

  /*!
   * Move a single node to a new cluster.
   *
   * @param u The node that is moved.
   * @param local_rand Thread-local \c Random object.
   * @param local_rating_map Thread-local rating map for gain computation.
   * @return Whether the node was moved to another cluster.
   */
  template <typename LocalRatingMap>
  bool handle_node(const NodeID u, Random &local_rand, LocalRatingMap &local_rating_map) {
    if (derived().skip_node(u)) {
      return false;
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = derived().cluster(u);
    const auto [new_cluster, new_gain] =
        find_best_cluster(u, u_weight, u_cluster, local_rand, local_rating_map);

    if (derived().cluster(u) != new_cluster) {
      if (derived().move_cluster_weight(
              u_cluster, new_cluster, u_weight, derived().max_cluster_weight(new_cluster)
          )) {
        derived().move_node(u, new_cluster);
        activate_neighbors(u);
        IFSTATS(_expected_total_gain += new_gain);

        return true;
      }
    }

    return false;
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
    ClusterID current_cluster;
    EdgeWeight current_gain;
    ClusterWeight current_cluster_weight;
  };

  /*!
   * Computes the best feasible cluster for a node.
   *
   * @param u The node for which the cluster is computed.
   * @param u_weight The weight of the node.
   * @param u_cluster The current cluster of the node.
   * @param local_rand Thread-local \c Random object.
   * @param local_rating_map Thread-local rating map to compute gain values.
   * @return Pair with: new cluster of the node, gain value for the move to the
   * new cluster.
   */
  template <typename LocalRatingMap>
  std::pair<ClusterID, EdgeWeight> find_best_cluster(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &local_rand,
      LocalRatingMap &local_rating_map
  ) {
    auto action = [&](auto &map) {
      const ClusterWeight initial_cluster_weight = derived().cluster_weight(u_cluster);
      ClusterSelectionState state{
          .local_rand = local_rand,
          .u = u,
          .u_weight = u_weight,
          .initial_cluster = u_cluster,
          .initial_cluster_weight = initial_cluster_weight,
          .best_cluster = u_cluster,
          .best_gain = 0,
          .best_cluster_weight = initial_cluster_weight,
          .current_cluster = 0,
          .current_gain = 0,
          .current_cluster_weight = 0,
      };

      bool is_interface_node = false;
      shm::lp::NeighborhoodRatings::accumulate(
          *_graph,
          u,
          map,
          [&](const NodeID v) { return derived().cluster(v); },
          [&](const NodeID v) {
            if (!derived().accept_neighbor(u, v)) {
              return false;
            }
            if (Config::kUseActiveSetStrategy && _use_local_active_set) {
              is_interface_node |= v >= _num_active_nodes;
            }
            return true;
          },
          _max_num_neighbors
      );

      if constexpr (Config::kUseActiveSetStrategy) {
        if (_use_active_set || (_use_local_active_set && !is_interface_node)) {
          _active.deactivate(u);
        }
      }

      const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        if constexpr (Config::kSkipLosingClusterWeights) {
          // Keep equal-gain tie-breaking and fresh weights for every potential winner.
          if (state.current_gain < state.best_gain) {
            continue;
          }
        }
        state.current_cluster_weight = derived().cluster_weight(cluster);

        if (derived().accept_cluster(state)) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      const EdgeWeight actual_gain = IFSTATS(state.best_gain - map[state.initial_cluster]);
      map.clear();
      return std::make_pair(state.best_cluster, actual_gain);
    };

    const auto [best_cluster, gain] = local_rating_map.execute(
        std::min<ClusterID>(_graph->degree(u), _initial_num_clusters), action
    );

    return {best_cluster, gain};
  }

  /*!
   * Flags neighbors of a node that has been moved as active.
   *
   * @param u Node that was moved.
   */
  void activate_neighbors(const NodeID u) {
    if (!Config::kUseActiveSetStrategy || (!_use_active_set && !_use_local_active_set)) {
      return;
    }

    _active.activate_neighbors(*_graph, u, [&](const NodeID v) {
      return derived().activate_neighbor(v);
    });
  }

  void activate_neighbors_of_ghost_node(const NodeID u) {
    KASSERT(_graph->is_ghost_node(u));

    if constexpr (!Config::kUseActiveSetStrategy) {
      return;
    }
    if (!_use_active_set) {
      return;
    }

    _graph->ghost_graph().adjacent_nodes(u, [&](const NodeID v) {
      if (derived().activate_neighbor(v)) {
        _active.activate(v);
      }
    });
  }

  NodeID perform_iteration(
      const NodeID from = 0,
      const NodeID to = std::numeric_limits<NodeID>::max(),
      const bool count_skipped_nodes = false
  ) {
    if (!_iteration_order_initialized || from != 0 || to != std::numeric_limits<NodeID>::max()) {
      _iteration_order.initialize(*_graph, _max_degree, from, to);
      _iteration_order_initialized = from == 0 && to == std::numeric_limits<NodeID>::max();
    }
    _count_skipped_nodes = count_skipped_nodes;
    _statistics.clear();
    auto kernel = shm::lp::make_single_phase_kernel(*this);
    _iteration_order.for_each(kernel);
    return _statistics.totals().moved;
  }

public: // Shared node-processing kernel interface.
  struct LocalWorker {
    RatingMap &ratings;
    Random &random;
    shm::lp::RoundStats &stats;
  };

  [[nodiscard]] bool should_stop() const {
    return false;
  }

  [[nodiscard]] bool should_stop(LocalWorker &, NodeID) const {
    return false;
  }

  [[nodiscard]] bool should_visit(const NodeID u) {
    if (_graph->degree(u) >= _max_degree) {
      return false;
    }
    if constexpr (Config::kUseActiveSetStrategy) {
      if ((_use_active_set || _use_local_active_set) && !_active.contains(u)) {
        if (_count_skipped_nodes) {
          ++_num_skipped_nodes_ets.local();
        }
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] LocalWorker make_local_worker(Random &random) {
    return {_rating_map_ets.local(), random, _statistics.local()};
  }

  void visit(const NodeID u, LocalWorker &local) {
    local.stats.moved += handle_node(u, local.random, local.ratings);
  }

  void finish_work_unit(LocalWorker &) {}

private:
  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
            if (Config::kUseActiveSetStrategy && (_use_active_set || _use_local_active_set)) {
              _active.initialize(u);
            }

            const ClusterID initial_cluster = derived().initial_cluster(u);
            derived().init_cluster(u, initial_cluster);
            derived().reset_node_state(u);
          });
        },
        [&] {
          tbb::parallel_for<ClusterID>(0, _initial_num_clusters, [&](const ClusterID cluster) {
            derived().init_cluster_weight(cluster, derived().initial_cluster_weight(cluster));
          });
        }
    );
    IFSTATS(_expected_total_gain = 0);
  }

private:
  Derived &derived() {
    return *static_cast<Derived *>(this);
  }

protected: // Default implementations
  [[nodiscard]] bool accept_cluster(const ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight <=
                derived().max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  void reset_node_state(const NodeID /* node */) {}

  [[nodiscard]] inline bool accept_neighbor(const NodeID /* u */, const NodeID /* v */) {
    return true;
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID /* node */) {
    return true;
  }

  [[nodiscard]] inline ClusterID initial_cluster(const NodeID u) {
    return derived().cluster(u);
  }

  [[nodiscard]] inline ClusterWeight initial_cluster_weight(const ClusterID cluster) {
    return derived().cluster_weight(cluster);
  }

  [[nodiscard]] inline bool skip_node(const NodeID /* node */) {
    return false;
  }

protected: // Members
  //! Graph we operate on, or \c nullptr if \c initialize has not been called
  //! yet.
  const Graph *_graph = nullptr;

  //! The number of non-empty clusters before we ran the first iteration of
  //! label propagation.
  ClusterID _initial_num_clusters;

  //! We do not move nodes with a degree higher than this. However, other nodes
  //! may still be moved to the cluster of with degree larger than this
  //! threshold.
  NodeID _max_degree = std::numeric_limits<NodeID>::max();

  //! When computing the gain values for a node, this is an upper limit on the
  //! number of neighbors of the nodes we consider. Any more neighbors are
  //! ignored.
  NodeID _max_num_neighbors = std::numeric_limits<NodeID>::max();

  //! Thread-local map to compute gain values.
  tbb::enumerable_thread_specific<RatingMap> _rating_map_ets;

  shm::lp::ActiveSet _active;

  //! If statistics are enabled, this is the sum of the gain of all moves that
  //! were performed. If executed single-thread, this should be equal to the
  //! reduction of the edge cut.
  parallel::Atomic<EdgeWeight> _expected_total_gain;

  tbb::enumerable_thread_specific<NodeID> _num_skipped_nodes_ets;

  bool _use_active_set = false;
  bool _use_local_active_set = false;

private:
  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
  ClusterID _prev_num_clusters = 0;
  shm::ChunkShuffledIterationOrder::Permutations _permutations;
  shm::ChunkShuffledIterationOrder _iteration_order{_permutations};
  bool _iteration_order_initialized = false;
  bool _count_skipped_nodes = false;
  shm::lp::RoundStatistics _statistics;
};

template <typename ClusterID, typename ClusterWeight> class OwnedRelaxedClusterWeightVector {
public:
  using ClusterWeights = StaticArray<ClusterWeight>;

  void allocate_cluster_weights(const ClusterID num_clusters) {
    if (_cluster_weights.size() < num_clusters) {
      _cluster_weights.resize(num_clusters);
    }
  }

  void setup_cluster_weights(ClusterWeights cluster_weights) {
    _cluster_weights = std::move(cluster_weights);
  }

  auto &&take_cluster_weights() {
    return std::move(_cluster_weights);
  }

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
    if (cluster_weight(new_cluster) + delta <= max_weight) {
      __atomic_fetch_add(&_cluster_weights[new_cluster], delta, __ATOMIC_RELAXED);
      __atomic_fetch_sub(&_cluster_weights[old_cluster], delta, __ATOMIC_RELAXED);
      return true;
    }
    return false;
  }

private:
  ClusterWeights _cluster_weights;
};

template <typename NodeID, typename ClusterID> class NonatomicClusterVectorRef {
public:
  void init_clusters_ref(StaticArray<ClusterID> &clustering) {
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

} // namespace kaminpar::dist
