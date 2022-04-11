/*******************************************************************************
 * @file:   i_clustering_algorithm.h
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Interface for clustering algorithms.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"
#include "kaminpar/parallel/atomic.h"

namespace dkaminpar {
template <typename ClusterID>
class IClustering {
public:
    using AtomicClusterArray = scalable_vector<shm::parallel::Atomic<ClusterID>>;

    virtual ~IClustering() = default;

    virtual const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, NodeWeight max_cluster_weight) = 0;
};
} // namespace dkaminpar
