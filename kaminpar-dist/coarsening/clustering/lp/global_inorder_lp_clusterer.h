/*******************************************************************************
 * Label propagation with clusters that can grow to multiple PEs.
 *
 * @file:   global_inorder_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
class GlobalInOrderLPClusterer : public Clusterer<GlobalNodeID> {
public:
  explicit GlobalInOrderLPClusterer(const Context &ctx);

  GlobalInOrderLPClusterer(const GlobalInOrderLPClusterer &) = delete;
  GlobalInOrderLPClusterer &operator=(const GlobalInOrderLPClusterer &) = delete;

  GlobalInOrderLPClusterer(GlobalInOrderLPClusterer &&) = default;
  GlobalInOrderLPClusterer &operator=(GlobalInOrderLPClusterer &&) = default;

  ~GlobalInOrderLPClusterer() override;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class GlobalInOrderLPClusteringImpl> _impl;
};
} // namespace kaminpar::dist
