/*******************************************************************************
 * @file:   noop_clustering.h
 * @author: Daniel Seemaier
 * @date:   13.05.2022
 * @brief:  Clustering algorithm that assigns each node to its own cluster.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"

namespace kaminpar::dist {
template <typename ClusterID>
class NoopClustering : public ClusteringAlgorithm<ClusterID> {
  using ClusterArray = typename ClusteringAlgorithm<ClusterID>::ClusterArray;

public:
  explicit NoopClustering(const Context &) {}

  ClusterArray &compute_clustering(const DistributedGraph &,
                                   const GlobalNodeWeight) final {
    return _empty_clustering;
  }

private:
  ClusterArray _empty_clustering;
};

using LocalNoopClustering = NoopClustering<NodeID>;
using GlobalNoopClustering = NoopClustering<GlobalNodeID>;
} // namespace kaminpar::dist
