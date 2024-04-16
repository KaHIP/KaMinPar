/*******************************************************************************
 * A dummy clusterer that assigns each node to its own singleton cluster.
 *
 * @file:   noop_clusterer.h
 * @author: Daniel Seemaier
 * @date:   16.06.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {
class NoopClusterer : public Clusterer {
public:
  NoopClusterer() = default;

  NoopClusterer(const NoopClusterer &) = delete;
  NoopClusterer &operator=(const NoopClusterer &) = delete;

  NoopClusterer(NoopClusterer &&) noexcept = default;
  NoopClusterer &operator=(NoopClusterer &&) noexcept = default;

  //
  // Optional options
  //

  virtual void set_max_cluster_weight(const NodeWeight /* weight */) {}
  virtual void set_desired_cluster_count(const NodeID /* count */) {}

  //
  // Clustering function
  //

  virtual void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, bool free_memory_afterwards
  ) {
    tbb::parallel_for<NodeID>(0, graph.n(), [&](const NodeID i) { clustering[i] = i; });
  }
};
} // namespace kaminpar::shm

