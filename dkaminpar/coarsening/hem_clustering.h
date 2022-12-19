/*******************************************************************************
 * @file:   hem_clustering.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 * @brief:  Clustering using heavy edge matching.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/definitions.h"

namespace kaminpar::dist {
class HEMClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
    HEMClustering(const Context& ctx);

    HEMClustering(const HEMClustering&)                = delete;
    HEMClustering& operator=(const HEMClustering&)     = delete;
    HEMClustering(HEMClustering&&) noexcept            = default;
    HEMClustering& operator=(HEMClustering&&) noexcept = default;

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    AtomicClusterArray _matching;
};
} // namespace kaminpar::dist
