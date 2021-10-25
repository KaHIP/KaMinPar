#pragma once

#include "dkaminpar/algorithm/local_graph_contraction.h"

namespace dkaminpar::graph {
namespace contraction {
struct GlobalMappingResult {
  DistributedGraph graph;
  scalable_vector<GlobalNodeID> mapping;
  MemoryContext m_ctx;
};
} // namespace contraction

contraction::GlobalMappingResult contract_global_clustering_redistribute(const DistributedGraph &graph,
                                                            const scalable_vector<GlobalNodeID> &clustering,
                                                            contraction::MemoryContext m_ctx = {});
} // namespace dkaminpar::graph