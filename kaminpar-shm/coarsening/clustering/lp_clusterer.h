/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class LPClustering : public Clusterer {
public:
  LPClustering(const CoarseningContext &c_ctx);

  LPClustering(const LPClustering &) = delete;
  LPClustering &operator=(const LPClustering &) = delete;

  LPClustering(LPClustering &&) noexcept = default;
  LPClustering &operator=(LPClustering &&) noexcept = default;

  ~LPClustering() override;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  void set_communities(std::span<const NodeID> communities) final;

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, bool free_memory_afterwards
  ) final;

private:
  std::unique_ptr<class LPClusteringImplWrapper> _impl_wrapper;
};

} // namespace kaminpar::shm
