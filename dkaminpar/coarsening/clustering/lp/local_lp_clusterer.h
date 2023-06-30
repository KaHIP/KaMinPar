/*******************************************************************************
 * Label propagation clustering that only clusters node within a PE (i.e., 
 * ignores ghost nodes).
 *
 * @file:   local_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class LocalLPClustering : public LocalClusterer {
public:
  explicit LocalLPClustering(const Context &ctx);

  LocalLPClustering(const LocalLPClustering &) = delete;
  LocalLPClustering &operator=(const LocalLPClustering &) = delete;

  LocalLPClustering(LocalLPClustering &&) = default;
  LocalLPClustering &operator=(LocalLPClustering &&) = default;

  ~LocalLPClustering() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class LocalLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
