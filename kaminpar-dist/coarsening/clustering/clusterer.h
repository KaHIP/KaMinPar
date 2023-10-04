/*******************************************************************************
 * Interface for clustering algorithms.
 *
 * @file:   clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.21
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/parallel/atomic.h"

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

class LocalClusterer : public Clusterer<NodeID> {
public:
  virtual ClusterArray &
  cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) = 0;

  virtual ClusterArray &
  cluster(const DistributedPartitionedGraph &p_graph, GlobalNodeWeight max_cluster_weight) = 0;
};
} // namespace kaminpar::dist
