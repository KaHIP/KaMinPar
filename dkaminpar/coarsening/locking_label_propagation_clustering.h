/*******************************************************************************
 * @file:   distributed_locking_label_propagation.h
 *
 * @author: Daniel Seemaier
 * @date:   01.10.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/i_clustering.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

namespace dkaminpar {
class LockingLabelPropagationClustering : public ClusteringAlgorithm {
public:
    explicit LockingLabelPropagationClustering(const Context& ctx);
    ~LockingLabelPropagationClustering();

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    std::unique_ptr<class LockingLabelPropagationClusteringImpl> _impl{};
};
} // namespace dkaminpar
