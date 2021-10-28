/*******************************************************************************
* @file:   locking_clustering_contraction.h
*
* @author: Daniel Seemaier
* @date:   25.10.2021
* @brief:  Contracts a clustering computed by \c LockingLabelPropagation.
******************************************************************************/
#pragma once

#include "../datastructure/distributed_graph.h"
#include "../distributed_definitions.h"
#include "local_graph_contraction.h"
#include "locking_label_propagation_clustering.h"

namespace dkaminpar::coarsening {
namespace contraction {
struct LockingClusteringContractionResult {
  DistributedGraph graph;
  scalable_vector<NodeID> mapping;
  MemoryContext m_ctx;
};
} // namespace contraction

contraction::LockingClusteringContractionResult
contract_locking_clustering(const DistributedGraph &graph, const LockingLpClustering::AtomicClusterArray &clustering,
                            contraction::MemoryContext m_ctx = {});
} // namespace dkaminpar::graph