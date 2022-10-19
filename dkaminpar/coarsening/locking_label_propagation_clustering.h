/*******************************************************************************
 * @file:   distributed_locking_label_propagation.h
 * @author: Daniel Seemaier
 * @date:   01.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"

namespace kaminpar::dist {
class LockingLabelPropagationClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
    explicit LockingLabelPropagationClustering(const Context& ctx);
    ~LockingLabelPropagationClustering();

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    std::unique_ptr<class LockingLabelPropagationClusteringImpl> _impl{};
};
} // namespace kaminpar::dist
