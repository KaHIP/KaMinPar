/*******************************************************************************
 * Clustering via heavy edge matching with label propagation fallback.
 *
 * @file:   hem_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
class HEMLPClusterer : public Clusterer {
public:
  HEMLPClusterer(const Context &ctx);

  HEMLPClusterer(const HEMLPClusterer &) = delete;
  HEMLPClusterer &operator=(const HEMLPClusterer &) = delete;

  HEMLPClusterer(HEMLPClusterer &&) noexcept = default;
  HEMLPClusterer &operator=(HEMLPClusterer &&) = delete;

  void set_max_cluster_weight(GlobalNodeWeight weight) final;

  void cluster(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) final;

private:
  GlobalNodeID compute_size_after_matching_contraction(const StaticArray<GlobalNodeID> &clustering);

  const DistributedGraph *_graph;

  bool _fallback = false;

  std::unique_ptr<Clusterer> _lp;
  std::unique_ptr<Clusterer> _hem;
};
} // namespace kaminpar::dist
