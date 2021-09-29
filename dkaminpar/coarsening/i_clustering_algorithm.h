/*******************************************************************************
 * @file:   i_clustering_algorithm.h
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Interface for clustering algorithms.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"

namespace dkaminpar {
template<typename ClusterID>
class ClusteringAlgorithm {
public:
  using AtomicClusterArray = scalable_vector<shm::parallel::IntegralAtomicWrapper<ClusterID>>;

  virtual const AtomicClusterArray &compute_clustering(const DistributedGraph &graph,
                                                       const NodeWeight max_cluster_weight) = 0;
};
} // namespace dkaminpar