/*******************************************************************************
 * @file:   distributed_local_label_propagation_clustering.h
 *
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:  Label propagation clustering that only clusters node within a PE
 * (i.e., not with ghost nodes).
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/i_clustering.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar {
class DistributedLocalLabelPropagationClustering : public ClusteringAlgorithm {
public:
    DistributedLocalLabelPropagationClustering(NodeID max_n, const CoarseningContext& c_ctx);

    DistributedLocalLabelPropagationClustering(const DistributedLocalLabelPropagationClustering&) = delete;
    DistributedLocalLabelPropagationClustering& operator=(const DistributedLocalLabelPropagationClustering&) = delete;
    DistributedLocalLabelPropagationClustering(DistributedLocalLabelPropagationClustering&&)                 = default;
    DistributedLocalLabelPropagationClustering& operator=(DistributedLocalLabelPropagationClustering&&) = default;

    ~DistributedLocalLabelPropagationClustering();

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    std::unique_ptr<class DistributedLocalLabelPropagationClusteringImpl> _impl;
};
} // namespace dkaminpar
