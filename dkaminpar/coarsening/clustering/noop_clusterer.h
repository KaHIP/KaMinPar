/*******************************************************************************
 * Pseudo-clusterer that assigns each node to its own cluster.
 *
 * @file:   noop_clusterer.h
 * @author: Daniel Seemaier
 * @date:   13.05.2022
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"

namespace kaminpar::dist {
template <typename ClusterID> class NoopClustering : public Clusterer<ClusterID> {
  using ClusterArray = typename Clusterer<ClusterID>::ClusterArray;

public:
  explicit NoopClustering(const Context &) {}

  void initialize(const DistributedGraph &) final {}

  ClusterArray &cluster(const DistributedGraph &, GlobalNodeWeight) final {
    return _empty_clustering;
  }

private:
  ClusterArray _empty_clustering;
};

using LocalNoopClustering = NoopClustering<NodeID>;
using GlobalNoopClustering = NoopClustering<GlobalNodeID>;
} // namespace kaminpar::dist
