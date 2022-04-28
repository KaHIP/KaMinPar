/*******************************************************************************
 * @file:   graph_extraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#include "dkaminpar/graphutils/graph_extraction.h"

#include "kaminpar/graphutils/graph_extraction.tcc"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/utils/math.h"

namespace dkaminpar::graph {
namespace {
using namespace shm::graph;

PEID compute_block_owner(const BlockID b, const BlockID k, const PEID num_pes) {
    return static_cast<PEID>(math::compute_local_range_rank<BlockID>(k, static_cast<BlockID>(num_pes), b));
}

SubgraphExtractionResult
extract_subgraphs(const DistributedPartitionedGraph& p_graph, SubgraphMemory& subgraph_memory) {
    return shm::graph::extract_subgraphs_impl(p_graph, subgraph_memory, [&](BlockID) { return 0; });
}
} // namespace

std::vector<DistributedGraph> distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph) {
    return {};
}
} // namespace dkaminpar::graph
