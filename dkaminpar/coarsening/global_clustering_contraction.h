/*******************************************************************************
 * @file:   global_clustering_contraction_redistribution.h
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Shared-memory parallel contraction of global clustering without
 * any restrictions.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "common/parallel/atomic.h"
#include "common/scalable_vector.h"

namespace kaminpar::dist {
using GlobalMapping    = scalable_vector<parallel::Atomic<GlobalNodeID>>;
using GlobalClustering = scalable_vector<parallel::Atomic<GlobalNodeID>>;

struct GlobalContractionResult {
    DistributedGraph graph;
    GlobalMapping    mapping;
};

GlobalContractionResult
contract_global_clustering_no_migration(const DistributedGraph& graph, const GlobalClustering& clustering);
GlobalContractionResult
contract_global_clustering_minimal_migration(const DistributedGraph& graph, const GlobalClustering& clustering);
GlobalContractionResult
contract_global_clustering_full_migration(const DistributedGraph& graph, const GlobalClustering& clustering);

GlobalContractionResult contract_global_clustering(
    const DistributedGraph& graph, const GlobalClustering& clustering, GlobalContractionAlgorithm algorithm
);

DistributedPartitionedGraph project_global_contracted_graph(
    const DistributedGraph& fine_graph, DistributedPartitionedGraph coarse_graph, const GlobalMapping& fine_to_coarse
);
} // namespace kaminpar::dist
