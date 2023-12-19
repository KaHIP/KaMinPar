/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clustering.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm {

template <typename Graph> class LPClusteringImpl;

class LPClustering : public Clusterer {
public:
  LPClustering(NodeID max_n, const CoarseningContext &c_ctx);

  LPClustering(const LPClustering &) = delete;
  LPClustering &operator=(const LPClustering &) = delete;

  LPClustering(LPClustering &&) noexcept = default;
  LPClustering &operator=(LPClustering &&) noexcept = default;

  ~LPClustering() override;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  const AtomicClusterArray &compute_clustering(const Graph &graph) final;

private:
  std::unique_ptr<LPClusteringImpl<CSRGraph>> _csr_core;
  std::unique_ptr<LPClusteringImpl<CompressedGraph>> _compressed_core;
};

} // namespace kaminpar::shm
