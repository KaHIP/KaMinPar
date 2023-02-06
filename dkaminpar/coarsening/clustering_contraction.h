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

struct ContractionResult {
    DistributedGraph           graph;
    NoinitVector<GlobalNodeID> mapping;
};

ContractionResult contract_clustering(const DistributedGraph& graph, const GlobalClustering& clustering);
} // namespace kaminpar::dist

