/*******************************************************************************
 * @file:   i_clustering_algorithm.h
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Interface for clustering algorithms.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "common/parallel/atomic.h"

namespace kaminpar::dist {
template <typename ClusterID>
class ClusteringAlgorithm {
public:
    using AtomicClusterArray = scalable_vector<parallel::Atomic<ClusterID>>;

    virtual ~ClusteringAlgorithm() = default;

    virtual const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) = 0;
};
} // namespace kaminpar::dist
