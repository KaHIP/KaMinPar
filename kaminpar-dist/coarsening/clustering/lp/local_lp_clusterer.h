/*******************************************************************************
 * Label propagation clustering that only clusters node within a PE (i.e.,
 * ignores ghost nodes).
 *
 * @file:   local_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
class LocalLPClusterer : public LocalClusterer {
public:
  explicit LocalLPClusterer(const Context &ctx);

  LocalLPClusterer(const LocalLPClusterer &) = delete;
  LocalLPClusterer &operator=(const LocalLPClusterer &) = delete;

  LocalLPClusterer(LocalLPClusterer &&) = default;
  LocalLPClusterer &operator=(LocalLPClusterer &&) = default;

  ~LocalLPClusterer() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

  ClusterArray &
  cluster(const DistributedPartitionedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class LocalLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
