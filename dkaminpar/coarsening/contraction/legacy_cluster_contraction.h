/*******************************************************************************
 * @file:   legacy_cluster_contraction_redistribution.h
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Shared-memory parallel contraction of global clustering without
 * any restrictions.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/contraction/cluster_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/definitions.h"

#include "common/parallel/atomic.h"
#include "common/scalable_vector.h"

namespace kaminpar::dist {
using LegacyGlobalMapping = scalable_vector<parallel::Atomic<GlobalNodeID>>;
using LegacyGlobalClustering = scalable_vector<parallel::Atomic<GlobalNodeID>>;

struct GlobalContractionResult {
  DistributedGraph graph;
  LegacyGlobalMapping mapping;
  MigratedNodes migration;
};

GlobalContractionResult contract_global_clustering_no_migration(
    const DistributedGraph &graph, const LegacyGlobalClustering &clustering
);
GlobalContractionResult contract_global_clustering_minimal_migration(
    const DistributedGraph &graph, const LegacyGlobalClustering &clustering
);
GlobalContractionResult contract_global_clustering_full_migration(
    const DistributedGraph &graph, const LegacyGlobalClustering &clustering
);

ContractionResult contract_global_clustering(
    const DistributedGraph &graph, GlobalClustering &clustering, const CoarseningContext &c_ctx
);

DistributedPartitionedGraph project_global_contracted_graph(
    const DistributedGraph &fine_graph,
    DistributedPartitionedGraph coarse_graph,
    const LegacyGlobalMapping &fine_to_coarse
);
} // namespace kaminpar::dist
