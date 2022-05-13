/*******************************************************************************
 * @file:   noop_clustering.h
 *
 * @author: Daniel Seemaier
 * @date:   13.05.2022
 * @brief:  Clustering algorithm that assigns each node to its own cluster.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/i_clustering_algorithm.h"
#include "dkaminpar/context.h"

namespace dkaminpar {
template <typename ClusterID>
class NoopClustering : public ClusteringAlgorithm<ClusterID> {
    using AtomicClusterArray = typename ClusteringAlgorithm<ClusterID>::AtomicClusterArray;

public:
    explicit NoopClustering(const Context&) {}

    const AtomicClusterArray& compute_clustering(const DistributedGraph&, const GlobalNodeWeight) final {
        return _empty_clustering;
    }

private:
    AtomicClusterArray _empty_clustering;
};

using LocalNoopClustering  = NoopClustering<NodeID>;
using GlobalNoopClustering = NoopClustering<GlobalNodeID>;
} // namespace dkaminpar
