/*******************************************************************************
 * Graph contraction for arbitrary clusterings.
 *
 * @file:   cluster_contraction.h
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 * @brief:  Graph contraction for arbitrary clusterings.
 ******************************************************************************/
#pragma once

#include <limits>
#include <vector>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

#include "common/datastructures/noinit_vector.h"

namespace kaminpar::dist {
using GlobalMapping = NoinitVector<GlobalNodeID>;
using GlobalClustering = NoinitVector<GlobalNodeID>;

struct MigratedNodes {
  NoinitVector<NodeID> nodes;

  std::vector<int> sendcounts;
  std::vector<int> sdispls;
  std::vector<int> recvcounts;
  std::vector<int> rdispls;
};

struct ContractionResult {
  DistributedGraph graph;
  NoinitVector<GlobalNodeID> mapping;
  MigratedNodes migration;
};

ContractionResult contract_clustering(
    const DistributedGraph &graph, GlobalClustering &clustering, const CoarseningContext &c_ctx
);

ContractionResult contract_clustering(
    const DistributedGraph &graph,
    GlobalClustering &clustering,
    double max_cnode_imbalance = std::numeric_limits<double>::max(),
    bool migrate_cnode_prefix = false,
    bool force_perfect_cnode_balance = true
);

DistributedPartitionedGraph project_partition(
    const DistributedGraph &graph,
    DistributedPartitionedGraph p_c_graph,
    const NoinitVector<GlobalNodeID> &c_mapping,
    const MigratedNodes &migration
);
} // namespace kaminpar::dist
