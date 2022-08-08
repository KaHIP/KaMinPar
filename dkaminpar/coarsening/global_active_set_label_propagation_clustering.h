/*******************************************************************************
 * @file:   global_active_set_label_propagation_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   25.05.2022
 * @brief:  Label propagation with clusters that span multiple PEs. Cluster
 * labels and weights are synchronized in rounds. Between communication rounds,
 * a cluster can grow beyond the maximum cluster weight limit if more than one
 * PE moves nodes to the cluster. Thus, the clustering might violate the
 * maximum cluster weight limit.
 * This implementation uses an active set strategy.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/i_clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace kaminpar::dist {
class DistributedActiveSetGlobalLabelPropagationClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
    DistributedActiveSetGlobalLabelPropagationClustering(const Context& ctx);

    DistributedActiveSetGlobalLabelPropagationClustering(const DistributedActiveSetGlobalLabelPropagationClustering&) =
        delete;
    DistributedActiveSetGlobalLabelPropagationClustering&
    operator=(const DistributedActiveSetGlobalLabelPropagationClustering&) = delete;

    DistributedActiveSetGlobalLabelPropagationClustering(DistributedActiveSetGlobalLabelPropagationClustering&&) =
        default;
    DistributedActiveSetGlobalLabelPropagationClustering&
    operator=(DistributedActiveSetGlobalLabelPropagationClustering&&) = default;

    ~DistributedActiveSetGlobalLabelPropagationClustering();

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    std::unique_ptr<class DistributedActiveSetGlobalLabelPropagationClusteringImpl> _impl;
};
} // namespace kaminpar::dist
