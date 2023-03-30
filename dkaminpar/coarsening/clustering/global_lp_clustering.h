/*******************************************************************************
 * @file:   global_lp_clustering.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 * @brief   Label propagation clustering without restrictions, i.e., clusters
 * can span across multiple PEs.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class GlobalLPClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
  explicit GlobalLPClustering(const Context &ctx);

  GlobalLPClustering(const GlobalLPClustering &) = delete;
  GlobalLPClustering &operator=(const GlobalLPClustering &) = delete;

  GlobalLPClustering(GlobalLPClustering &&) = default;
  GlobalLPClustering &operator=(GlobalLPClustering &&) = default;

  ~GlobalLPClustering() override;

  ClusterArray &compute_clustering(
      const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight
  ) final;

private:
  std::unique_ptr<class GlobalLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
