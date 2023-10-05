/*******************************************************************************
 * Pseudo-clusterer that assigns each node to its own cluster.
 *
 * @file:   noop_clusterer.h
 * @author: Daniel Seemaier
 * @date:   13.05.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/context.h"

namespace kaminpar::dist {
class GlobalNoopClustering : public Clusterer<GlobalNodeID> {
  using ClusterArray = typename Clusterer<GlobalNodeID>::ClusterArray;

public:
  explicit GlobalNoopClustering(const Context &) {}

  void initialize(const DistributedGraph &) final {}

  ClusterArray &cluster(const DistributedGraph &, GlobalNodeWeight) final {
    return _empty_clustering;
  }

protected:
  ClusterArray _empty_clustering;
};

class LocalNoopClustering : public LocalClusterer {
  using ClusterArray = typename Clusterer<NodeID>::ClusterArray;

public:
  explicit LocalNoopClustering(const Context &) {}

  void initialize(const DistributedGraph &) final {}

  ClusterArray &cluster(const DistributedGraph &, GlobalNodeWeight) final {
    return _empty_clustering;
  }

  ClusterArray &cluster(const DistributedPartitionedGraph &, GlobalNodeWeight) final {
    return _empty_clustering;
  }

private:
  ClusterArray _empty_clustering;
};
} // namespace kaminpar::dist
