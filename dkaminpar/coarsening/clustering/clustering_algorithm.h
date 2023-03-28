/*******************************************************************************
 * @file:   clustering_algorithm.h
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Interface for clustering algorithms.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "common/parallel/atomic.h"

namespace kaminpar::dist {
template <typename ClusterID> class ClusteringAlgorithm {
public:
  using ClusterArray = NoinitVector<ClusterID>;

  virtual ~ClusteringAlgorithm() = default;

  virtual ClusterArray &compute_clustering(
      const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight
  ) = 0;
};
} // namespace kaminpar::dist
