/*******************************************************************************
 * Graph contraction for arbitrary clusterings.
 *
 * @file:   cluster_contraction.h
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 ******************************************************************************/
#pragma once

#include <limits>
#include <vector>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
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
 * Stores technical mappings necessary to project a partition of the coarse graph to the fine graph.
 * Part of the contraction result and should not be used outside the `project_partition()` function.
 */
struct MigratedNodes {
  StaticArray<NodeID> nodes;

  std::vector<int> sendcounts;
  std::vector<int> sdispls;
  std::vector<int> recvcounts;
  std::vector<int> rdispls;
};

/**
 * Stores the contracted graph along with information necessary to project a partition of the coarse
 * graph to the fine graph.
 */
struct ContractionResult {
  DistributedGraph graph;
  StaticArray<GlobalNodeID> mapping;
  MigratedNodes migration;
};

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
ContractionResult contract_clustering(
    const DistributedGraph &graph,
    StaticArray<GlobalNodeID> &clustering,
    const CoarseningContext &c_ctx
);

/**
 * Constructs the coarse graph given a clustering of the fine graph.
 *
 * @param graph The fine graph.
 * @param clustering The clustering of the fine graph: this is an array of size `graph.total_n()`
 * (i.e., one entry for each owned node *and* ghost node). The assignment of ghost nodes must be
 * consistent with their assignment on other PEs. Cluster IDs can be arbitrary integers in the range
 * `0 <= ID < graph.global_n()`.
 * @param max_cnode_imbalance The maximum allowed imbalance of coarse nodes (per PE). If a PE would
 * end up with too many coarse nodes, the contraction algorithm will move coarse nodes to rebalance
 * the assignment.
 * @param migrate_cnode_prefix If `true`, the contraction algorithm will move a prefix of coarse
 * nodes if their assignment violates the maximum allowed imbalance factor; otherwise, it moves a
 * suffix.
 * @param force_perfect_cnode_balance If `true`, the contraction algorithm will perfectly balance
 * the coarse node assignment if their natural assignment would violate the given imbalance factor.
 *
 * @return The coarse graph along with information necessary to project a partition of the coarse to
 * the fine graph.
 */
ContractionResult contract_clustering(
    const DistributedGraph &graph,
    StaticArray<GlobalNodeID> &clustering,
    double max_cnode_imbalance = std::numeric_limits<double>::max(),
    bool migrate_cnode_prefix = false,
    bool force_perfect_cnode_balance = true
);

/**
 * Projects the partition of a coarse graph back onto the fine graph.
 *
 * @param graph The fine graph.
 * @param p_c_graph The partition of the coarse graph.
 * @param c_mapping The mapping from coarse nodes to fine nodes (part of `ContractionResult`).
 * @param migration The migration information for coarse nodes (part of `ContractionResult`).
 *
 * @return The partition of the fine graph.
 */
DistributedPartitionedGraph project_partition(
    const DistributedGraph &graph,
    DistributedPartitionedGraph p_c_graph,
    const StaticArray<GlobalNodeID> &c_mapping,
    const MigratedNodes &migration
);
} // namespace kaminpar::dist
