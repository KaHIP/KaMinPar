/*******************************************************************************
 * @file:   parallel_label_propagation_clustering.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 * @brief:  Parallel label propgation for clustering.
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/clusterer.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"

namespace kaminpar::shm {
class LPClustering : public Clusterer {
public:
  LPClustering(
      NodeID max_n, const CoarseningContext &c_ctx
  );

  LPClustering(const LPClustering &) = delete;
  LPClustering &operator=(const LPClustering &) = delete;

  LPClustering(LPClustering &&) noexcept = default;
  LPClustering &operator=(LPClustering &&) noexcept = default;

  ~LPClustering() override;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  const AtomicClusterArray &compute_clustering(const Graph &graph) final;

private:
  std::unique_ptr<class LPClusteringImpl> _core;
};

} // namespace kaminpar::shm
