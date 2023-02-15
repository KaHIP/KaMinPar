/*******************************************************************************
 * @file:   local_label_propagation_clustering.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 * @brief:  Label propagation clustering that only clusters node within a PE
 * (i.e., not with ghost nodes).
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class DistributedLocalLabelPropagationClustering : public ClusteringAlgorithm<NodeID> {
public:
    DistributedLocalLabelPropagationClustering(const Context& ctx);

    DistributedLocalLabelPropagationClustering(const DistributedLocalLabelPropagationClustering&)            = delete;
    DistributedLocalLabelPropagationClustering& operator=(const DistributedLocalLabelPropagationClustering&) = delete;

    DistributedLocalLabelPropagationClustering(DistributedLocalLabelPropagationClustering&&)            = default;
    DistributedLocalLabelPropagationClustering& operator=(DistributedLocalLabelPropagationClustering&&) = default;

    ~DistributedLocalLabelPropagationClustering();

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    std::unique_ptr<class DistributedLocalLabelPropagationClusteringImpl> _impl;
};
} // namespace kaminpar::dist
