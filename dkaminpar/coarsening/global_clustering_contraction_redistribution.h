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

DistributedPartitionedGraph project_global_contracted_graph(const DistributedGraph &fine_graph,
                                                            DistributedPartitionedGraph coarse_graph,
                                                            const GlobalMapping &fine_to_coarse);
} // namespace dkaminpar::coarsening