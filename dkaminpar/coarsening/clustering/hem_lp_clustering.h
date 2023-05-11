/*******************************************************************************
 * @file:   hem_lp_clustering.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 * @brief:  Clustering using heavy edge matching and label propagation.
 ******************************************************************************/
#pragma once

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/definitions.h"

namespace kaminpar::dist {
class HEMLPClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
  HEMLPClustering(const Context &ctx);

  HEMLPClustering(const HEMLPClustering &) = delete;
  HEMLPClustering &operator=(const HEMLPClustering &) = delete;

  HEMLPClustering(HEMLPClustering &&) noexcept = default;
  HEMLPClustering &operator=(HEMLPClustering &&) = delete;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  GlobalNodeID compute_size_after_matching_contraction(const ClusterArray &clustering);

  const DistributedGraph *_graph;
  bool _fallback = false;

  std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>> _lp;
  std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>> _hem;
};
} // namespace kaminpar::dist
