/*******************************************************************************
 * Interface for clustering algorithms.
 *
 * @file:   clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.21
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/dkaminpar.h"

#include "common/parallel/atomic.h"

namespace kaminpar::dist {
template <typename ClusterID> class Clusterer {
public:
  using ClusterArray = NoinitVector<ClusterID>;

  virtual ~Clusterer() = default;

  virtual void initialize(const DistributedGraph &graph) = 0;

  virtual ClusterArray &
  cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) = 0;
};

using GlobalClusterer = Clusterer<GlobalNodeID>;
using LocalClusterer = Clusterer<NodeID>;
} // namespace kaminpar::dist
