/*******************************************************************************
 * @file:   clustering_contraction.h
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 * @brief:  Graph contraction for arbitrary clusterings.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"

#include "common/parallel/atomic.h"

namespace kaminpar::dist {
using GlobalClustering = scalable_vector<parallel::Atomic<GlobalNodeID>>;

struct MigratedNodes {
    NoinitVector<NodeID> nodes;

    std::vector<int> sendcounts;
    std::vector<int> sdispls;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
};

struct ContractionResult {
    DistributedGraph           graph;
    NoinitVector<GlobalNodeID> mapping;
    MigratedNodes              migration;
};

ContractionResult contract_clustering(const DistributedGraph& graph, const GlobalClustering& clustering);

DistributedPartitionedGraph project_partition(
    const DistributedGraph& graph, DistributedPartitionedGraph p_c_graph, const NoinitVector<GlobalNodeID>& c_mapping,
    const MigratedNodes& migration
);
} // namespace kaminpar::dist

