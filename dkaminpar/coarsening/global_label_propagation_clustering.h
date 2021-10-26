/*******************************************************************************
* @file:   global_label_propagation_coarsener.h
*
* @author: Daniel Seemaier
* @date:   29.09.21
* @brief:  Label propagation across PEs, synchronizes cluster weights after
* every weight, otherwise moves nodes without communication causing violations
* of the balance constraint.
******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/i_clustering.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_context.h"

namespace dkaminpar {
class DistributedGlobalLabelPropagationClustering : public IClustering<GlobalNodeID> {
public:
  DistributedGlobalLabelPropagationClustering(NodeID max_n, const CoarseningContext &c_ctx);

  DistributedGlobalLabelPropagationClustering(const DistributedGlobalLabelPropagationClustering &) = delete;
  DistributedGlobalLabelPropagationClustering &operator=(const DistributedGlobalLabelPropagationClustering &) = delete;
  DistributedGlobalLabelPropagationClustering(DistributedGlobalLabelPropagationClustering &&) = default;
  DistributedGlobalLabelPropagationClustering &operator=(DistributedGlobalLabelPropagationClustering &&) = default;

  ~DistributedGlobalLabelPropagationClustering();

  const AtomicClusterArray &compute_clustering(const DistributedGraph &graph, NodeWeight max_cluster_weight) final;

private:
  std::unique_ptr<class DistributedGlobalLabelPropagationClusteringImpl> _impl;
};
} // namespace dkaminpar