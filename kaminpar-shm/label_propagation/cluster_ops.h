/*******************************************************************************
 * Cluster storage and weight management helpers for label propagation.
 *
 * @file:   cluster_ops.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar {

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
    ClusterWeights new_cluster_weights(num_new_clusters);

    tbb::parallel_for(
        tbb::blocked_range<ClusterID>(0, _cluster_weights.size()), [&](const auto &r) {
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
