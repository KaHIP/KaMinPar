/*******************************************************************************
 * Clustering adapter that overlays an ensemble of clusterings.
 *
 * @file:   ensemble_clusterer.h
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/coarsening/clusterer.h"

namespace kaminpar::shm {

StaticArray<NodeID> overlay_clusterings(
    const Graph &graph, StaticArray<NodeID> first, const StaticArray<NodeID> &second
);

class EnsembleClusterer final : public Clusterer {
public:
  EnsembleClusterer(std::unique_ptr<Clusterer> clusterer, std::size_t num_clusterings);

  void set_num_clusterings(std::size_t num_clusterings);

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;
  void set_communities(std::span<const NodeID> communities) final;

  void compute_clustering(
      StaticArray<NodeID> &clustering, const Graph &graph, bool free_memory_afterwards
  ) final;

private:
  std::unique_ptr<Clusterer> _clusterer;
  std::size_t _num_clusterings;
};

} // namespace kaminpar::shm
