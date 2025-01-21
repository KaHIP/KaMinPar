/*******************************************************************************
 * Interface for clustering algorithms used for coarsening.
 *
 * @file:   clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include <span>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class Clusterer {
public:
  Clusterer() = default;

  Clusterer(const Clusterer &) = delete;
  Clusterer &operator=(const Clusterer &) = delete;

  Clusterer(Clusterer &&) noexcept = default;
  Clusterer &operator=(Clusterer &&) noexcept = default;

  virtual ~Clusterer() = default;

  //
  // Optional options
  //

  virtual void set_max_cluster_weight(NodeWeight /* weight */) {}
  virtual void set_desired_cluster_count(NodeID /* count */) {}

  virtual void set_communities(std::span<const NodeID> /* communities */) {}

  //
  // Clustering function
  //

  virtual void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, bool free_memory_afterwards
  ) = 0;
};

} // namespace kaminpar::shm
