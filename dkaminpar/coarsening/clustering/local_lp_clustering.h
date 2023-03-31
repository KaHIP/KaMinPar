/*******************************************************************************
 * @file:   local_label_propagation_clustering.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 * @brief:  Label propagation clustering that only clusters node within a PE
 * (i.e., not with ghost nodes).
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class LocalLPClustering : public ClusteringAlgorithm<NodeID> {
public:
  explicit LocalLPClustering(const Context &ctx);

  LocalLPClustering(const LocalLPClustering &) = delete;
  LocalLPClustering &operator=(const LocalLPClustering &) = delete;

  LocalLPClustering(LocalLPClustering &&) = default;
  LocalLPClustering &operator=(LocalLPClustering &&) = default;

  ~LocalLPClustering() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(
      const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight
  ) final;

private:
  std::unique_ptr<class LocalLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
