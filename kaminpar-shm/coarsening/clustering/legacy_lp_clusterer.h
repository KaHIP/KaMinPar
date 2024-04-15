/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   legacy_lp_clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm {
class LegacyLPClustering : public Clusterer {
public:
  LegacyLPClustering(NodeID max_n, const CoarseningContext &c_ctx);

  LegacyLPClustering(const LegacyLPClustering &) = delete;
  LegacyLPClustering &operator=(const LegacyLPClustering &) = delete;

  LegacyLPClustering(LegacyLPClustering &&) noexcept = default;
  LegacyLPClustering &operator=(LegacyLPClustering &&) noexcept = default;

  ~LegacyLPClustering() override;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, bool free_memory_afterwards
  ) final;

private:
  std::unique_ptr<class LegacyLPClusteringImpl> _core;
};
} // namespace kaminpar::shm
