/*******************************************************************************
 * Label propagation with clusters that can grow to multiple PEs.
 *
 * @file:   global_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

namespace kaminpar::dist {

class GlobalLPClusterer : public Clusterer {
public:
  explicit GlobalLPClusterer(const Context &ctx);

  GlobalLPClusterer(const GlobalLPClusterer &) = delete;
  GlobalLPClusterer &operator=(const GlobalLPClusterer &) = delete;

  GlobalLPClusterer(GlobalLPClusterer &&) = default;
  GlobalLPClusterer &operator=(GlobalLPClusterer &&) = default;

  ~GlobalLPClusterer() override;

  void set_max_cluster_weight(GlobalNodeWeight weight) final;

  void cluster(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) final;

private:
  std::unique_ptr<class GlobalLPClusteringImplWrapper> _impl;
};

} // namespace kaminpar::dist
