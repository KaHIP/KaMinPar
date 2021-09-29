/*******************************************************************************
 * @file:   parallel_label_propagation_clustering.h
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Parallel label propgation for clustering.
 ******************************************************************************/
#pragma once

#include "kaminpar/coarsening/i_clustering_algorithm.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/parallel.h"

namespace kaminpar {
class LabelPropagationClusteringAlgorithm : public IClusteringAlgorithm {
public:
  LabelPropagationClusteringAlgorithm(NodeID max_n, const CoarseningContext &c_ctx);
  ~LabelPropagationClusteringAlgorithm() override;

  LabelPropagationClusteringAlgorithm(const LabelPropagationClusteringAlgorithm &) = delete;
  LabelPropagationClusteringAlgorithm &operator=(const LabelPropagationClusteringAlgorithm &) = delete;
  LabelPropagationClusteringAlgorithm(LabelPropagationClusteringAlgorithm &&) noexcept = default;
  LabelPropagationClusteringAlgorithm &operator=(LabelPropagationClusteringAlgorithm &&) noexcept = default;

  void set_max_cluster_weight(NodeWeight max_cluster_weight) final;
  void set_desired_cluster_count(NodeID count) final;

  const AtomicClusterArray &compute_clustering(const Graph &graph) final;

private:
  std::unique_ptr<class LabelPropagationClusteringCore> _core;
};

} // namespace kaminpar
