/*******************************************************************************
 * Interface for clustering algorithms used for coarsening.
 *
 * @file:   clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm {
class Clusterer {
public:
  using AtomicClusterArray = scalable_vector<parallel::Atomic<NodeID>>;

  Clusterer() = default;

  Clusterer(const Clusterer &) = delete;
  Clusterer &operator=(const Clusterer &) = delete;

  Clusterer(Clusterer &&) noexcept = default;
  Clusterer &operator=(Clusterer &&) noexcept = default;

  virtual ~Clusterer() = default;

  //
  // Optional options
  //

  virtual void set_max_cluster_weight(const NodeWeight /* weight */) {}
  virtual void set_desired_cluster_count(const NodeID /* count */) {}

  //
  // Clustering function
  //

  virtual const AtomicClusterArray &compute_clustering(const Graph &graph) = 0;
};
} // namespace kaminpar::shm
