/*******************************************************************************
 * Clustering via heavy edge matching with label propagation fallback.
 *
 * @file:   hem_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#pragma once

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/coarsening/clustering/clusterer.h"
#include "dkaminpar/context.h"
#include "dkaminpar/dkaminpar.h"

namespace kaminpar::dist {
class HEMLPClusterer : public GlobalClusterer {
public:
  HEMLPClusterer(const Context &ctx);

  HEMLPClusterer(const HEMLPClusterer &) = delete;
  HEMLPClusterer &operator=(const HEMLPClusterer &) = delete;

  HEMLPClusterer(HEMLPClusterer &&) noexcept = default;
  HEMLPClusterer &operator=(HEMLPClusterer &&) = delete;

  void initialize(const DistributedGraph &graph) final;

  ClusterArray &cluster(const DistributedGraph &graph, GlobalNodeWeight max_cluster_weight) final;

private:
  GlobalNodeID compute_size_after_matching_contraction(const ClusterArray &clustering);

  const DistributedGraph *_graph;
  bool _fallback = false;

  std::unique_ptr<GlobalClusterer> _lp;
  std::unique_ptr<GlobalClusterer> _hem;
};
} // namespace kaminpar::dist
