/*******************************************************************************
 * Mutable labels and cluster weights for label-propagation clustering.
 *
 * @file:   clustering_state.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <span>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/algorithms/label_propagation/active_set.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm::lp {

struct MoveResult {
  bool moved = false;
  bool emptied_cluster = false;
};

/*!
 * Run-scoped direct access to clustering state used by the node hot path.
 *
 * The owning ClusteringState may resize its arrays between runs. A view is
 * therefore created after reset() and only retained for the current run.
 */
class ClusteringStateView {
public:
  ClusteringStateView(
      NodeID *clustering,
      NodeWeight *cluster_weights,
      NodeID *favored_clusters,
      ActiveSetView active,
      const NodeWeight max_cluster_weight,
      const std::span<const NodeID> communities
  )
      : _clustering(clustering),
        _cluster_weights(cluster_weights),
        _favored_clusters(favored_clusters),
        _active(active),
        _max_cluster_weight(max_cluster_weight),
        _communities(communities) {}

  [[nodiscard]] KAMINPAR_INLINE NodeID cluster(const NodeID u) const {
    return __atomic_load_n(&_clustering[u], __ATOMIC_RELAXED);
  }

  KAMINPAR_INLINE void move_node(const NodeID u, const NodeID cluster) {
    __atomic_store_n(&_clustering[u], cluster, __ATOMIC_RELAXED);
  }

  [[nodiscard]] KAMINPAR_INLINE NodeWeight cluster_weight(const NodeID cluster) const {
    return __atomic_load_n(&_cluster_weights[cluster], __ATOMIC_RELAXED);
  }

  [[nodiscard]] KAMINPAR_INLINE NodeWeight max_cluster_weight(const NodeID) const {
    return _max_cluster_weight;
  }

  [[nodiscard]] KAMINPAR_INLINE bool has_communities() const {
    return !_communities.empty();
  }

  [[nodiscard]] KAMINPAR_INLINE bool accepts_community(const NodeID from, const NodeID to) const {
    return _communities.empty() || _communities[from] == _communities[to];
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  accepts_community_unchecked(const NodeID from, const NodeID to) const {
    return _communities[from] == _communities[to];
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  move_cluster_weight(const NodeID from, const NodeID to, const NodeWeight delta) {
    if (cluster_weight(to) + delta <= _max_cluster_weight) {
      __atomic_fetch_add(&_cluster_weights[to], delta, __ATOMIC_RELAXED);
      __atomic_fetch_sub(&_cluster_weights[from], delta, __ATOMIC_RELAXED);
      return true;
    }
    return false;
  }

  template <bool kParallelActivate = false, typename Graph>
  KAMINPAR_INLINE MoveResult commit(
      const Graph &graph,
      const NodeID u,
      const NodeID from,
      const NodeID to,
      const NodeWeight u_weight
  ) {
    if (from == to || cluster_weight(to) + u_weight > _max_cluster_weight) {
      return {};
    }

    __atomic_fetch_add(&_cluster_weights[to], u_weight, __ATOMIC_RELAXED);
    const NodeWeight prev_from =
        __atomic_fetch_sub(&_cluster_weights[from], u_weight, __ATOMIC_RELAXED);

    move_node(u, to);
    if constexpr (kParallelActivate) {
      _active.activate_neighbors_parallel(graph, u, [](const NodeID) { return true; });
    } else {
      _active.activate_neighbors(graph, u, [](const NodeID) { return true; });
    }
    return {.moved = true, .emptied_cluster = prev_from == u_weight};
  }

  [[nodiscard]] KAMINPAR_INLINE bool is_active(const NodeID u) const {
    return _active.contains(u);
  }

  KAMINPAR_INLINE void deactivate(const NodeID u) {
    _active.deactivate(u);
  }

  KAMINPAR_INLINE void set_favored_cluster(const NodeID u, const NodeID cluster) {
    _favored_clusters[u] = cluster;
  }

private:
  NodeID *_clustering;
  NodeWeight *_cluster_weights;
  NodeID *_favored_clusters;
  ActiveSetView _active;
  NodeWeight _max_cluster_weight;
  std::span<const NodeID> _communities;
};

class ClusteringState {
public:
  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  void set_desired_num_clusters(const NodeID desired_num_clusters) {
    _desired_num_clusters = desired_num_clusters;
  }

  void set_communities(const std::span<const NodeID> communities) {
    _communities = communities;
  }

  template <typename Graph> void reset(StaticArray<NodeID> &clustering, const Graph &graph) {
    const NodeID n = graph.n();
    _clustering = &clustering;
    if (_cluster_weights.size() < n) {
      _cluster_weights.resize(n, static_array::noinit);
    }
    if (_favored_clusters.size() < n) {
      _favored_clusters.resize(n, static_array::noinit);
    }
    _active.resize(n);

    NodeID *__restrict__ clustering_ptr = clustering.data();
    NodeID *__restrict__ favored_ptr = _favored_clusters.data();
    NodeWeight *__restrict__ weights_ptr = _cluster_weights.data();
    if (graph.is_node_weighted()) {
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &range) {
        for (NodeID u = range.begin(); u != range.end(); ++u) {
          _active.initialize(u);
          clustering_ptr[u] = u;
          favored_ptr[u] = u;
          weights_ptr[u] = graph.node_weight(u);
        }
      });
    } else {
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &range) {
        for (NodeID u = range.begin(); u != range.end(); ++u) {
          _active.initialize(u);
          clustering_ptr[u] = u;
          favored_ptr[u] = u;
          weights_ptr[u] = 1;
        }
      });
    }

    _current_num_clusters = n;
  }

  [[nodiscard]] NodeID cluster(const NodeID u) const {
    return __atomic_load_n(&_clustering->at(u), __ATOMIC_RELAXED);
  }

  void move_node(const NodeID u, const NodeID cluster) {
    __atomic_store_n(&_clustering->at(u), cluster, __ATOMIC_RELAXED);
  }

  [[nodiscard]] NodeWeight cluster_weight(const NodeID cluster) const {
    return __atomic_load_n(&_cluster_weights[cluster], __ATOMIC_RELAXED);
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) const {
    return _max_cluster_weight;
  }

  [[nodiscard]] bool
  move_cluster_weight(const NodeID from, const NodeID to, const NodeWeight delta) {
    if (cluster_weight(to) + delta <= _max_cluster_weight) {
      __atomic_fetch_add(&_cluster_weights[to], delta, __ATOMIC_RELAXED);
      __atomic_fetch_sub(&_cluster_weights[from], delta, __ATOMIC_RELAXED);
      return true;
    }
    return false;
  }

  void set_favored_cluster(const NodeID u, const NodeID cluster) {
    _favored_clusters[u] = cluster;
  }

  [[nodiscard]] NodeID favored_cluster(const NodeID u) const {
    return __atomic_load_n(&_favored_clusters[u], __ATOMIC_RELAXED);
  }

  [[nodiscard]] bool
  compare_exchange_favored_cluster(const NodeID u, NodeID &expected, const NodeID desired) {
    return __atomic_compare_exchange_n(
        &_favored_clusters[u], &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
    );
  }

  void remove_empty_clusters(const NodeID count) {
    _current_num_clusters -= count;
  }

  [[nodiscard]] bool should_stop() const {
    return _current_num_clusters <= _desired_num_clusters;
  }

  [[nodiscard]] NodeID num_clusters() const {
    return _current_num_clusters;
  }

  [[nodiscard]] ClusteringStateView view() {
    return ClusteringStateView(
        _clustering->data(),
        _cluster_weights.data(),
        _favored_clusters.data(),
        _active.view(),
        _max_cluster_weight,
        _communities
    );
  }

  void free() {
    _active.free();
    _cluster_weights.free();
    _favored_clusters.free();
    _clustering = nullptr;
  }

private:
  StaticArray<NodeID> *_clustering = nullptr;
  StaticArray<NodeWeight> _cluster_weights;
  StaticArray<NodeID> _favored_clusters;
  ActiveSet _active;

  parallel::Atomic<NodeID> _current_num_clusters;
  NodeID _desired_num_clusters = 0;
  NodeWeight _max_cluster_weight = kInvalidNodeWeight;
  std::span<const NodeID> _communities;
};

} // namespace kaminpar::shm::lp
