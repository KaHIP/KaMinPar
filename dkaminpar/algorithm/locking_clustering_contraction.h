/*******************************************************************************
* @file:   locking_clustering_contraction.h
*
* @author: Daniel Seemaier
* @date:   25.10.2021
* @brief:  Contracts a clustering computed by \c LockingLabelPropagation.
******************************************************************************/
#pragma once

#include "dkaminpar/algorithm/local_graph_contraction.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"

namespace dkaminpar::graph {
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