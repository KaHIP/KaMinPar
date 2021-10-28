/*******************************************************************************
 * @file:   seq_global_clustering_contraction_redistribution.h
 *
 * @author: Daniel Seemaier
 * @date:   26.10.2021
 * @brief:  Sequential code to contract a global clustering without any
 * limitations and redistribute the contracted graph such that each PE gets
 * an equal number of edges.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/local_graph_contraction.h"

namespace dkaminpar::coarsening {
namespace contraction {
struct GlobalMappingResult {
  DistributedGraph graph{};
  scalable_vector<GlobalNodeID> mapping{};
  MemoryContext m_ctx{};
};
} // namespace contraction

contraction::GlobalMappingResult contract_global_clustering_redistribute_sequential(
    const DistributedGraph &graph,
    const scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>> &clustering,
    contraction::MemoryContext m_ctx = {});
} // namespace dkaminpar::coarsening