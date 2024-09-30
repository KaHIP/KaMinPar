/*******************************************************************************
 * Label propagation clustering that only clusters node within a PE (i.e.,
 * ignores ghost nodes).
 *
 * @file:   local_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

namespace kaminpar::dist {

class LocalLPClusterer : public Clusterer {
public:
  explicit LocalLPClusterer(const Context &ctx);

  LocalLPClusterer(const LocalLPClusterer &) = delete;
  LocalLPClusterer &operator=(const LocalLPClusterer &) = delete;

  LocalLPClusterer(LocalLPClusterer &&) = default;
  LocalLPClusterer &operator=(LocalLPClusterer &&) = default;

  ~LocalLPClusterer() override;

  void set_communities(const StaticArray<BlockID> &communities) final;
  void clear_communities() final;

  void set_max_cluster_weight(GlobalNodeWeight weight) final;

  void cluster(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) final;

private:
  std::unique_ptr<class LocalLPClusteringImplWrapper> _impl;
};

} // namespace kaminpar::dist
