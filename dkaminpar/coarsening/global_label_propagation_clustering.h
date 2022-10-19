/*******************************************************************************
 * @file:   global_label_propagation_coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Label propagation with clusters that span multiple PEs. Cluster
 * labels and weights are synchronized in rounds. Between communication rounds,
 * a cluster can grow beyond the maximum cluster weight limit if more than one
 * PE moves nodes to the cluster. Thus, the clustering might violate the
 * maximum cluster weight limit.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class DistributedGlobalLabelPropagationClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
    DistributedGlobalLabelPropagationClustering(const Context& ctx);

    DistributedGlobalLabelPropagationClustering(const DistributedGlobalLabelPropagationClustering&)            = delete;
    DistributedGlobalLabelPropagationClustering& operator=(const DistributedGlobalLabelPropagationClustering&) = delete;

    DistributedGlobalLabelPropagationClustering(DistributedGlobalLabelPropagationClustering&&)            = default;
    DistributedGlobalLabelPropagationClustering& operator=(DistributedGlobalLabelPropagationClustering&&) = default;

    ~DistributedGlobalLabelPropagationClustering();

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    std::unique_ptr<class DistributedGlobalLabelPropagationClusteringImpl> _impl;
};
} // namespace kaminpar::dist
