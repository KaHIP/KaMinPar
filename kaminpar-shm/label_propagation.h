/*******************************************************************************
 * Generic implementation of parallel label propagation.
 *
 * @file:   parallel_label_propagation.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <atomic>
#include <type_traits>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/scalable_allocator.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/tags.h"
#include "kaminpar-common/timer.h"

namespace kaminpar {
struct LabelPropagationConfig {
  using Graph = ::kaminpar::shm::Graph;

  // Data structure used to accumulate edge weights for gain value calculation
  using RatingMap =
      ::kaminpar::RatingMap<shm::EdgeWeight, shm::NodeID, FastResetArray<shm::EdgeWeight>>;

  // Data type for cluster IDs and weights
  using ClusterID = tag::Mandatory;
  using ClusterWeight = tag::Mandatory;

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

  // If true, we count the number of empty clusters
  static constexpr bool kTrackClusterCount = false;

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
template <typename Derived, typename Config> class LabelPropagation {
  static_assert(std::is_base_of_v<LabelPropagationConfig, Config>);

  SET_DEBUG(false);
  SET_STATISTICS(false);

protected:
  using RatingMap = typename Config::RatingMap;
  using Graph = typename Config::Graph;
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using ClusterID = typename Config::ClusterID;
  using ClusterWeight = typename Config::ClusterWeight;

public:
  void set_max_degree(const NodeID max_degree) {
    _max_degree = max_degree;
  }
  [[nodiscard]] NodeID max_degree() const {
    return _max_degree;
  }

  void set_max_num_neighbors(const ClusterID max_num_neighbors) {
    _max_num_neighbors = max_num_neighbors;
  }
  [[nodiscard]] ClusterID max_num_neighbors() const {
    return _max_num_neighbors;
  }

  void set_desired_num_clusters(const ClusterID desired_num_clusters) {
    _desired_num_clusters = desired_num_clusters;
  }
  [[nodiscard]] ClusterID desired_num_clusters() const {
    return _desired_num_clusters;
  }

  [[nodiscard]] EdgeWeight expected_total_gain() const {
    return _expected_total_gain;
  }

protected:
  /*!
   * (Re)allocates memory to run label propagation on a graph with \c num_nodes
   * nodes.
   * @param num_nodes Number of nodes in the graph.
   */
  void allocate(const NodeID num_nodes, const ClusterID num_clusters) {
    allocate(num_nodes, num_nodes, num_clusters);
  }

  /*!
   * (Re)allocates memory to run label propagation on a graph with \c num_nodes
   * nodes in total, but a clustering is only computed for the first \c
   * num_active_nodes nodes.
   *
   * This is mostly useful for distributed graphs where ghost nodes are always
   * inactive.
   *
   * @param num_nodes Total number of nodes in the graph, i.e., neighbors of
   * active nodes have an ID less than this.
   * @param num_active_nodes Number of nodes for which a cluster label is
   * computed.
   */
  void allocate(const NodeID num_nodes, const NodeID num_active_nodes, const NodeID num_clusters) {
    if (_num_nodes < num_nodes) {
      if constexpr (Config::kUseLocalActiveSetStrategy) {
        _active.resize(num_nodes);
      }
      _num_nodes = num_nodes;
    }

    if (_num_active_nodes < num_active_nodes) {
      if constexpr (Config::kUseActiveSetStrategy) {
        _active.resize(num_active_nodes);
      }
      if constexpr (Config::kUseTwoHopClustering) {
        _favored_clusters.resize(num_active_nodes);
      }
      _num_active_nodes = num_active_nodes;
    }
    if (_num_clusters < num_clusters) {
      for (auto &rating_map : _rating_map_ets) {
        rating_map.change_max_size(num_clusters);
      }
      _num_clusters = num_clusters;
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
        graph->n() == 0 || (_num_nodes > 0u && _num_active_nodes > 0u),
        "you must call allocate() before initialize()"
    );

    _graph = graph;
    _initial_num_clusters = num_clusters;
    _current_num_clusters = num_clusters;
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
   * Move a single node to a new cluster.
   *
   * @param u The node that is moved.
   * @param local_rand Thread-local \c Random object.
   * @param local_rating_map Thread-local rating map for gain computation.
   * @return Pair with: whether the node was moved to another cluster, whether
   * the previous cluster is now empty.
   */
  template <typename LocalRatingMap>
  std::pair<bool, bool>
  handle_node(const NodeID u, Random &local_rand, LocalRatingMap &local_rating_map) {
    if (derived_skip_node(u)) {
      return {false, false};
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    const ClusterID u_cluster = derived_cluster(u);
    const auto [new_cluster, new_gain] =
        find_best_cluster(u, u_weight, u_cluster, local_rand, local_rating_map);

    if (derived_cluster(u) != new_cluster) {
      if (derived_move_cluster_weight(
              u_cluster, new_cluster, u_weight, derived_max_cluster_weight(new_cluster)
          )) {
        derived_move_node(u, new_cluster);
        activate_neighbors(u);
        IFSTATS(_expected_total_gain += new_gain);

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
      const ClusterWeight initial_cluster_weight = derived_cluster_weight(u_cluster);
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

      auto add_to_rating_map = [&](const EdgeID e, const NodeID v) {
        if (derived_accept_neighbor(u, v)) {
          const ClusterID v_cluster = derived_cluster(v);
          const EdgeWeight rating = _graph->edge_weight(e);
          map[v_cluster] += rating;
          if constexpr (Config::kUseLocalActiveSetStrategy) {
            is_interface_node |= v >= _num_active_nodes;
          }
        }
      };

      const EdgeID from = _graph->first_edge(u);
      const EdgeID to = from + std::min(_graph->degree(u), _max_num_neighbors);
      for (EdgeID e = from; e < to; ++e) {
        add_to_rating_map(e, _graph->edge_target(e));
      }

      if constexpr (Config::kUseLocalActiveSetStrategy) {
        if (!is_interface_node) {
          _active[u] = 0;
        }
      }
      if constexpr (Config::kUseActiveSetStrategy) {
        _active[u] = 0;
      }

      // After LP, we might want to use 2-hop clustering to merge nodes that
      // could not find any cluster to join for this, we store a favored cluster
      // for each node u if:
      // (1) we actually use 2-hop clustering
      // (2) u is still in a singleton cluster (weight of node == weight of cluster)
      // (3) the cluster is light (at most half full)
      ClusterID favored_cluster = u_cluster;
      const bool store_favored_cluster =
          Config::kUseTwoHopClustering && u_weight == initial_cluster_weight &&
          initial_cluster_weight <= derived_max_cluster_weight(u_cluster) / 2;

      const EdgeWeight gain_delta = (Config::kUseActualGain) ? map[u_cluster] : 0;

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = derived_cluster_weight(cluster);

        if (store_favored_cluster && state.current_gain > state.best_gain) {
          favored_cluster = state.current_cluster;
        }

        if (derived_accept_cluster(state)) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      // if we couldn't join any cluster, we store the favored cluster
      if (store_favored_cluster && state.best_cluster == state.initial_cluster) {
        _favored_clusters[u] = favored_cluster;
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
    for (const NodeID v : _graph->adjacent_nodes(u)) {
      // call derived_activate_neighbor() even if we do not use the active set
      // strategy since the function might have side effects; the compiler
      // should remove it if it does not side effects
      if (derived_activate_neighbor(v)) {
        if constexpr (Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy) {
          _active[v].store(1, std::memory_order_relaxed);
        }
      }
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

    auto handle_node = [&](DynamicFlatMap<ClusterID, NodeID> &matching_map, const NodeID u) {
      ClusterID &rep_key = matching_map[_favored_clusters[u]];

      if (rep_key == 0) {
        rep_key = u + 1;
      } else {
        const ClusterID rep = rep_key - 1;

        const bool could_move_u_to_rep = derived_move_cluster_weight(
            u, rep, derived_cluster_weight(u), derived_max_cluster_weight(rep)
        );

        if constexpr (match) {
          KASSERT(could_move_u_to_rep);
          derived_move_node(u, rep);
          rep_key = 0;
        } else {
          if (could_move_u_to_rep) {
            derived_move_node(u, rep);
          } else {
            rep_key = u + 1;
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

      const NodeID C = _favored_clusters[u];
      auto &sync = _favored_clusters[C];

      do {
        NodeID cluster = sync;

        if (cluster == C) {
          if (sync.compare_exchange_strong(cluster, u)) {
            // We are done: other nodes will join our cluster
            break;
          }
          if (cluster == C) {
            continue;
          }
        }

        // Invariant: cluster is a node with favored cluster C
        KASSERT(
            _favored_clusters[cluster] == C,
            "invariant violated by: " << V(u) << V(cluster) << V(C) << V(_favored_clusters[C])
        );

        // Try to join the cluster:
        if constexpr (match) {
          // Matching mode: try to build a cluster only containing nodes "cluster" and "u"
          if (sync.compare_exchange_strong(cluster, C)) {
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

            // We are done: joined cluster "cluster"
            break;
          } else if (sync.compare_exchange_strong(cluster, u)) {
            // We are done: other nodes will join our cluster
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

  //! Return the maximum weight of cluster \c cluster.
  [[nodiscard]] ClusterWeight derived_max_cluster_weight(const ClusterID cluster) {
    return static_cast<Derived *>(this)->max_cluster_weight(cluster);
  }

  //! Determine whether a node should be moved to a new cluster.
  [[nodiscard]] bool derived_accept_cluster(const ClusterSelectionState &state) {
    return static_cast<Derived *>(this)->accept_cluster(state);
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
  //! Graph we operate on, or \c nullptr if \c initialize has not been called
  //! yet.
  const Graph *_graph{nullptr};

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

  //! Thread-local map to compute gain values.
  tbb::enumerable_thread_specific<RatingMap> _rating_map_ets{[this] {
    return RatingMap(_num_clusters);
  }};

  //! Flags nodes with at least one node in its neighborhood that changed
  //! clusters during the last iteration. Nodes without this flag set must not
  //! be considered in the next iteration.
  scalable_vector<parallel::Atomic<uint8_t>> _active;

  //! If a node cannot join any cluster during an iteration, this vector stores
  //! the node's highest rated cluster independent of the maximum cluster
  //! weight. This information is used during 2-hop clustering.
  scalable_vector<parallel::Atomic<ClusterID>> _favored_clusters;

  //! If statistics are enabled, this is the sum of the gain of all moves that
  //! were performed. If executed single-thread, this should be equal to the
  //! reduction of the edge cut.
  parallel::Atomic<EdgeWeight> _expected_total_gain;

private:
  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
};

/*!
 * Parallel label propagation template that iterates over nodes in their natural
 * order.
 * @tparam Derived Derived subclass for static polymorphism.
 * @tparam Config Algorithmic configuration and data types.
 */
template <typename Derived, typename Config>
class InOrderLabelPropagation : public LabelPropagation<Derived, Config> {
  static_assert(std::is_base_of_v<LabelPropagationConfig, Config>);
  SET_DEBUG(true);

protected:
  using Base = LabelPropagation<Derived, Config>;

  using Graph = typename Base::Graph;
  using ClusterID = typename Base::ClusterID;
  using ClusterWeight = typename Base::ClusterWeight;
  using EdgeID = typename Base::EdgeID;
  using EdgeWeight = typename Base::EdgeWeight;
  using NodeID = typename Base::NodeID;
  using NodeWeight = typename Base::NodeWeight;

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
              if (!_active[u].load(std::memory_order_relaxed)) {
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

/*!
 * Parallel label propagation template that iterates over nodes in chunk random
 * order.
 * @tparam Derived Derived subclass for static polymorphism.
 * @tparam Config Algorithmic configuration and data types.
 */
template <typename Derived, typename Config>
class ChunkRandomdLabelPropagation : public LabelPropagation<Derived, Config> {
  using Base = LabelPropagation<Derived, Config>;
  static_assert(std::is_base_of_v<LabelPropagationConfig, Config>);

  SET_DEBUG(false);

protected:
  using Graph = typename Base::Graph;
  using ClusterID = typename Base::ClusterID;
  using ClusterWeight = typename Base::ClusterWeight;
  using EdgeID = typename Base::EdgeID;
  using EdgeWeight = typename Base::EdgeWeight;
  using NodeID = typename Base::NodeID;
  using NodeWeight = typename Base::NodeWeight;

  using Base::handle_node;
  using Base::set_max_degree;
  using Base::set_max_num_neighbors;
  using Base::should_stop;

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
   * LabelPropagationConfig::kMinChunkSize, which is a lower bound on the sum of
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

    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;
    parallel::Atomic<std::size_t> next_chunk = 0;

    tbb::parallel_for(static_cast<std::size_t>(0), _chunks.size(), [&](const std::size_t) {
      if (should_stop()) {
        return;
      }

      auto &local_num_moved_nodes = num_moved_nodes_ets.local();
      auto &local_rand = Random::instance();
      auto &local_rating_map = _rating_map_ets.local();
      NodeID num_removed_clusters = 0;

      const auto chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
      const auto &chunk = _chunks[chunk_id];
      const auto &permutation = _random_permutations.get(local_rand);

      const std::size_t num_sub_chunks =
          std::ceil(1.0 * (chunk.end - chunk.start) / Config::kPermutationSize);
      std::vector<NodeID> sub_chunk_permutation(num_sub_chunks);
      std::iota(sub_chunk_permutation.begin(), sub_chunk_permutation.end(), 0);
      local_rand.shuffle(sub_chunk_permutation);

      for (std::size_t sub_chunk = 0; sub_chunk < num_sub_chunks; ++sub_chunk) {
        for (std::size_t i = 0; i < Config::kPermutationSize; ++i) {
          const NodeID u = chunk.start +
                           Config::kPermutationSize * sub_chunk_permutation[sub_chunk] +
                           permutation[i % Config::kPermutationSize];
          if (u < chunk.end && _graph->degree(u) < _max_degree &&
              ((!Config::kUseActiveSetStrategy && !Config::kUseLocalActiveSetStrategy) ||
               _active[u].load(std::memory_order_relaxed))) {
            const auto [moved_node, emptied_cluster] = handle_node(u, local_rand, local_rating_map);
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

    return num_moved_nodes_ets.combine(std::plus{});
  }

private:
  struct Chunk {
    NodeID start;
    NodeID end;
  };

  struct Bucket {
    std::size_t start;
    std::size_t end;
  };

  void shuffle_chunks() {
    tbb::parallel_for<std::size_t>(0, _buckets.size(), [&](const std::size_t i) {
      const auto &bucket = _buckets[i];
      Random::instance().shuffle(_chunks.begin() + bucket.start, _chunks.begin() + bucket.end);
    });
  }

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
      tbb::enumerable_thread_specific<std::size_t> num_chunks_ets;
      tbb::enumerable_thread_specific<std::vector<Chunk>> chunks_ets;

      const std::size_t bucket_start = std::max(_graph->first_node_in_bucket(bucket), from);

      tbb::parallel_for(
          static_cast<int>(0),
          tbb::this_task_arena::max_concurrency(),
          [&](const int) {
            auto &chunks = chunks_ets.local();
            auto &num_chunks = num_chunks_ets.local();

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

      const std::size_t num_chunks = num_chunks_ets.combine(std::plus{});

      const std::size_t chunks_start = _chunks.size();
      parallel::Atomic<std::size_t> pos = chunks_start;
      _chunks.resize(chunks_start + num_chunks);
      tbb::parallel_for(chunks_ets.range(), [&](auto &r) {
        for (auto &chunk : r) {
          const std::size_t local_pos = pos.fetch_add(chunk.size());
          std::copy(chunk.begin(), chunk.end(), _chunks.begin() + local_pos);
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
            KASSERT(
                _graph->degree(u) == 0u || hit[u],
                V(_graph->degree(u)) << V(from) << V(u + from) << V(to)
            );
          }

          return true;
        }(),
        "",
        assert::heavy
    );
  }

protected:
  using Base::_active;
  using Base::_current_num_clusters;
  using Base::_graph;
  using Base::_max_degree;
  using Base::_rating_map_ets;

  RandomPermutations<NodeID, Config::kPermutationSize, Config::kNumberOfNodePermutations>
      _random_permutations{};
  std::vector<Chunk> _chunks;
  std::vector<Bucket> _buckets;
};

template <typename NodeID, typename ClusterID> class NonatomicOwnedClusterVector {
public:
  explicit NonatomicOwnedClusterVector(const NodeID max_num_nodes) : _clusters(max_num_nodes) {
    tbb::parallel_for<NodeID>(0, max_num_nodes, [&](const NodeID u) { _clusters[u] = 0; });
  }

  [[nodiscard]] auto &&take_clusters() {
    return std::move(_clusters);
  }

  [[nodiscard]] auto &clusters() {
    return _clusters;
  }

  void init_cluster(const NodeID node, const ClusterID cluster) {
    move_node(node, cluster);
  }

  [[nodiscard]] ClusterID cluster(const NodeID node) {
    KASSERT(node < _clusters.size());
    return __atomic_load_n(&_clusters[node], __ATOMIC_RELAXED);
  }

  void move_node(const NodeID node, const ClusterID cluster) {
    KASSERT(node < _clusters.size());
    __atomic_store_n(&_clusters[node], cluster, __ATOMIC_RELAXED);
  }

  void ensure_cluster_size(const NodeID max_num_nodes) {
    if (_clusters.size() < max_num_nodes) {
      _clusters.resize(max_num_nodes);
    }
  }

private:
  NoinitVector<ClusterID> _clusters;
};

template <typename NodeID, typename ClusterID> class OwnedClusterVector {
public:
  explicit OwnedClusterVector(const NodeID max_num_nodes) : _clusters(max_num_nodes) {}

  [[nodiscard]] auto &&take_clusters() {
    return std::move(_clusters);
  }

  [[nodiscard]] auto &clusters() {
    return _clusters;
  }

  void init_cluster(const NodeID node, const ClusterID cluster) {
    _clusters[node] = cluster;
  }

  [[nodiscard]] ClusterID cluster(const NodeID node) {
    KASSERT(node < _clusters.size());
    return _clusters[node];
  }

  void move_node(const NodeID node, const ClusterID cluster) {
    KASSERT(node < _clusters.size());
    _clusters[node] = cluster;
  }

  void ensure_cluster_size(const NodeID max_num_nodes) {
    if (_clusters.size() < max_num_nodes) {
      _clusters.resize(max_num_nodes);
    }
  }

private:
  scalable_vector<parallel::Atomic<ClusterID>> _clusters;
};

template <typename ClusterID, typename ClusterWeight> class OwnedRelaxedClusterWeightVector {
public:
  explicit OwnedRelaxedClusterWeightVector(const ClusterID max_num_clusters)
      : _cluster_weights(max_num_clusters) {}

  auto &&take_cluster_weights() {
    return std::move(_cluster_weights);
  }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights[cluster] = weight;
  }

  ClusterWeight cluster_weight(const ClusterID cluster) {
    return _cluster_weights[cluster];
  }

  bool move_cluster_weight(
      const ClusterID old_cluster,
      const ClusterID new_cluster,
      const ClusterWeight delta,
      const ClusterWeight max_weight
  ) {
    if (_cluster_weights[new_cluster] + delta <= max_weight) {
      _cluster_weights[new_cluster].fetch_add(delta, std::memory_order_relaxed);
      _cluster_weights[old_cluster].fetch_sub(delta, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

private:
  scalable_vector<parallel::Atomic<ClusterWeight>> _cluster_weights;
};
} // namespace kaminpar
