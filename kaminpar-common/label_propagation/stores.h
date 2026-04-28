/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   stores.h
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::lp {

template <typename NodeID, typename ClusterID> class ExternalLabelArray {
public:
  using ClusterIDType = ClusterID;

  void init(StaticArray<ClusterID> &labels) {
    _labels = &labels;
  }

  void init_cluster(const NodeID node, const ClusterID cluster) {
    move_node(node, cluster);
  }

  [[nodiscard]] ClusterID cluster(const NodeID node) const {
    KASSERT(_labels != nullptr);
    KASSERT(node < _labels->size());
    return __atomic_load_n(&_labels->at(node), __ATOMIC_RELAXED);
  }

  void move_node(const NodeID node, const ClusterID cluster) {
    KASSERT(_labels != nullptr);
    KASSERT(node < _labels->size());
    __atomic_store_n(&_labels->at(node), cluster, __ATOMIC_RELAXED);
  }

  [[nodiscard]] ClusterID initial_cluster(const NodeID node) const {
    return node;
  }

private:
  StaticArray<ClusterID> *_labels = nullptr;
};

template <typename ClusterID, typename ClusterWeight> class RelaxedClusterWeightVector {
public:
  using ClusterWeightType = ClusterWeight;
  using ClusterWeights = StaticArray<ClusterWeight>;

  void allocate(const ClusterID num_clusters) {
    if (_cluster_weights.size() < num_clusters) {
      _cluster_weights.resize(num_clusters);
    }
  }

  void free() {
    _cluster_weights.free();
  }

  void setup(ClusterWeights cluster_weights) {
    _cluster_weights = std::move(cluster_weights);
  }

  ClusterWeights release() {
    return std::move(_cluster_weights);
  }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights[cluster] = weight;
  }

  [[nodiscard]] ClusterWeight cluster_weight(const ClusterID cluster) const {
    return __atomic_load_n(&_cluster_weights[cluster], __ATOMIC_RELAXED);
  }

  [[nodiscard]] ClusterWeight initial_cluster_weight(const ClusterID cluster) const {
    return cluster_weight(cluster);
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
          for (ClusterID c = r.begin(); c != r.end(); ++c) {
            const ClusterWeight weight = _cluster_weights[c];

            if (weight != 0) {
              const ClusterID new_cluster = mapping[c] - 1;
              new_cluster_weights[new_cluster] = weight;
            }
          }
        }
    );

    _cluster_weights = std::move(new_cluster_weights);
  }

private:
  ClusterWeights _cluster_weights;
};

} // namespace kaminpar::lp
