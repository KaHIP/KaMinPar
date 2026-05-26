/*******************************************************************************
 * Heavy edge matching for graph coarsening / clustering.
 *
 * @file:   hem_clusterer.h
 ******************************************************************************/
#pragma once

#include <span>

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class HEMClustering : public Clusterer {
public:
  explicit HEMClustering(const CoarseningContext &c_ctx);

  HEMClustering(const HEMClustering &) = delete;
  HEMClustering &operator=(const HEMClustering &) = delete;

  HEMClustering(HEMClustering &&) noexcept = default;
  HEMClustering &operator=(HEMClustering &&) noexcept = default;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  void set_communities(std::span<const NodeID> communities) final;

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, bool free_memory_afterwards
  ) final;

private:
  NodeWeight _max_cluster_weight = kInvalidNodeWeight;
  NodeID _desired_cluster_count = 0;

  std::span<const NodeID> _communities;
};

} // namespace kaminpar::shm
