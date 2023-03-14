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
class LabelPropagationClusteringAlgorithm : public Clusterer {
public:
  LabelPropagationClusteringAlgorithm(NodeID max_n,
                                      const CoarseningContext &c_ctx);
  ~LabelPropagationClusteringAlgorithm() override;

  LabelPropagationClusteringAlgorithm(
      const LabelPropagationClusteringAlgorithm &) = delete;
  LabelPropagationClusteringAlgorithm &
  operator=(const LabelPropagationClusteringAlgorithm &) = delete;
  LabelPropagationClusteringAlgorithm(
      LabelPropagationClusteringAlgorithm &&) noexcept = default;
  LabelPropagationClusteringAlgorithm &
  operator=(LabelPropagationClusteringAlgorithm &&) noexcept = default;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  const AtomicClusterArray &compute_clustering(const Graph &graph) final;

private:
  std::unique_ptr<class LabelPropagationClusteringCore> _core;
};

} // namespace kaminpar::shm
