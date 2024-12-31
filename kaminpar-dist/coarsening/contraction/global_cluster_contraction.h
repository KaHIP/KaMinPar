/*******************************************************************************
 * Graph contraction for arbitrary clusterings.
 *
 * @file:   global_cluster_contraction.h
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 ******************************************************************************/
#pragma once

#include <limits>

#include "kaminpar-dist/coarsening/contraction.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {

namespace debug {

/**
 * Validates the given clustering, i.e., whether it is a valid input to the `contract_clustering()`
 * function.
 *
 * @param graph The graph for which the clustering was computed.
 * @param lnode_to_gcluster The clustering to validate.
 *
 * @return `true` if the clustering is valid, `false` otherwise. If `false` is returned, calling
 * `contract_clustering()` with the same clustering is undefined behavior.
 */
bool validate_clustering(
    const DistributedGraph &graph, const StaticArray<GlobalNodeID> &lnode_to_gcluster
);

} // namespace debug

/**
 * Constructs the coarse graph given a clustering of the fine graph.
 *
 * @param graph The fine graph.
 * @param clustering The clustering of the fine graph: this is an array of size `graph.total_n()`
 * (i.e., one entry for each owned node *and* ghost node). The assignment of ghost nodes must be
 * consistent with their assignment on other PEs. Cluster IDs can be arbitrary integers in the range
 * `0 <= ID < graph.global_n()`.
 * @param c_ctx Coarsening context with configuration parameters that influence the distribution of
 * the contracted graph.
 *
 * @return The coarse graph along with information necessary to project a partition of the coarse to
 * the fine graph.
 */
std::unique_ptr<CoarseGraph> contract_clustering(
    const DistributedGraph &graph,
    StaticArray<GlobalNodeID> &clustering,
    const CoarseningContext &c_ctx
);

std::unique_ptr<CoarseGraph> contract_clustering(
    const DistributedGraph &graph,
    StaticArray<GlobalNodeID> &clustering,
    double max_cnode_imbalance = std::numeric_limits<double>::max(),
    double max_cedge_imbalance = std::numeric_limits<double>::max(),
    bool migrate_cnode_prefix = false,
    bool strict_rebalancing = true
);

} // namespace kaminpar::dist
