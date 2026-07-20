/*******************************************************************************
 * Clustering adapter that overlays an ensemble of clusterings.
 *
 * @file:   ensemble_clusterer.cc
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/ensemble_clusterer.h"

#include <algorithm>
#include <stdexcept>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

StaticArray<NodeID> overlay_clusterings(
    const Graph &graph, StaticArray<NodeID> first, const StaticArray<NodeID> &second
) {
  const NodeID n = graph.n();
  KASSERT(first.size() >= n && second.size() >= n, "clustering arrays are too small");

  if (n == 0) {
    return first;
  }

  StaticArray<NodeID> index, buckets, leader_mapping;
  contraction::fill_leader_mapping(graph, first, leader_mapping);
  const NodeID c_n = leader_mapping[n - 1];
  auto mapping = contraction::compute_mapping(graph, std::move(first), leader_mapping);
  contraction::fill_cluster_buckets(c_n, graph, mapping, index, buckets);

  tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID c_u) {
    std::sort(
        buckets.begin() + index[c_u],
        buckets.begin() + index[c_u + 1],
        [&](const NodeID u, const NodeID v) { return second[u] < second[v]; }
    );

    NodeID previous_cluster = kInvalidNodeID;
    NodeID current_id = index[c_u] - 1;
    for (std::size_t i = index[c_u]; i < index[c_u + 1]; ++i) {
      const NodeID u = buckets[i];
      if (second[u] != previous_cluster) {
        ++current_id;
      }
      mapping[u] = current_id;
      previous_cluster = second[u];
    }
  });

  return mapping;
}

EnsembleClusterer::EnsembleClusterer(
    std::unique_ptr<Clusterer> clusterer, const std::size_t num_clusterings
)
    : _clusterer(std::move(clusterer)),
      _num_clusterings(num_clusterings) {
  if (_clusterer == nullptr) {
    throw std::invalid_argument("ensemble clusterer requires an underlying clusterer");
  }
  set_num_clusterings(num_clusterings);
}

void EnsembleClusterer::set_num_clusterings(const std::size_t num_clusterings) {
  if (num_clusterings == 0) {
    throw std::invalid_argument("ensemble clustering requires at least one clustering");
  }
  _num_clusterings = num_clusterings;
}

void EnsembleClusterer::set_max_cluster_weight(const NodeWeight max_cluster_weight) {
  _clusterer->set_max_cluster_weight(max_cluster_weight);
}

void EnsembleClusterer::set_desired_cluster_count(const NodeID count) {
  _clusterer->set_desired_cluster_count(count);
}

void EnsembleClusterer::set_communities(const std::span<const NodeID> communities) {
  _clusterer->set_communities(communities);
}

void EnsembleClusterer::compute_clustering(
    StaticArray<NodeID> &clustering, const Graph &graph, const bool free_memory_afterwards
) {
  _clusterer->compute_clustering(
      clustering, graph, free_memory_afterwards && _num_clusterings == 1
  );

  for (std::size_t i = 1; i < _num_clusterings; ++i) {
    StaticArray<NodeID> next_clustering(graph.n(), static_array::noinit);
    const bool last_clustering = i + 1 == _num_clusterings;
    _clusterer->compute_clustering(
        next_clustering, graph, free_memory_afterwards && last_clustering
    );
    clustering = overlay_clusterings(graph, std::move(clustering), next_clustering);
  }
}

} // namespace kaminpar::shm
