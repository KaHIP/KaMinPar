/*******************************************************************************
 * @file:   locking_clustering_contraction.h
 *
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Contracts a clustering computed by \c LockingLabelPropagation.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/coarsening.h"
#include "dkaminpar/coarsening/local_graph_contraction.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"

namespace dkaminpar::coarsening {
namespace contraction {
struct SparseClusteringContractionResult {
  DistributedGraph graph;
  GlobalMapping mapping;
};
} // namespace contraction

contraction::SparseClusteringContractionResult contract_clustering_sparse(const DistributedGraph &graph,
                                                                          const GlobalClustering &clustering);
} // namespace dkaminpar::coarsening