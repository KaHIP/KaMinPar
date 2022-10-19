/*******************************************************************************
 * @file:   clustering_algorithm.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 * @brief:  Interface for clustering algorithms used for coarsening.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"

#include "common/parallel/atomic.h"

namespace kaminpar::shm {
class ClusteringAlgorithm {
public:
    using AtomicClusterArray = scalable_vector<parallel::Atomic<NodeID>>;

    ClusteringAlgorithm()          = default;
    virtual ~ClusteringAlgorithm() = default;

    ClusteringAlgorithm(const ClusteringAlgorithm&)                = delete;
    ClusteringAlgorithm& operator=(const ClusteringAlgorithm&)     = delete;
    ClusteringAlgorithm(ClusteringAlgorithm&&) noexcept            = default;
    ClusteringAlgorithm& operator=(ClusteringAlgorithm&&) noexcept = default;

    //
    // Optional options
    //

    virtual void set_max_cluster_weight(const NodeWeight /* weight */) {}
    virtual void set_desired_cluster_count(const NodeID /* count */) {}

    //
    // Clustering function
    //

    virtual const AtomicClusterArray& compute_clustering(const Graph& graph) = 0;
};
} // namespace kaminpar::shm
