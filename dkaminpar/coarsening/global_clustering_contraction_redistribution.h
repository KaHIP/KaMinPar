/*******************************************************************************
 * @file:   global_clustering_contraction_redistribution.h
 *
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Shared-memory parallel contraction of global clustering without
 * any restrictions.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/coarsening.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar::coarsening {
struct RedistributedGlobalContractionResult {
  DistributedGraph graph;
  GlobalMapping mapping{};
};

RedistributedGlobalContractionResult contract_global_clustering_redistribute(const DistributedGraph &graph,
                                                                             const GlobalClustering &clustering);
} // namespace dkaminpar::coarsening