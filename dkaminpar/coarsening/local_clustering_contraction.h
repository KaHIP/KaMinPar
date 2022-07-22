/*******************************************************************************
 * @file:   local_graph_contraction.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "common/datastructures/ts_navigable_linked_list.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar::coarsening {
namespace contraction {
struct Edge {
    NodeID     target;
    EdgeWeight weight;
};

struct MemoryContext {
    scalable_vector<NodeID>                                               buckets;
    scalable_vector<shm::parallel::Atomic<NodeID>>                        buckets_index;
    scalable_vector<shm::parallel::Atomic<NodeID>>                        leader_mapping;
    scalable_vector<shm::NavigationMarker<NodeID, Edge, scalable_vector>> all_buffered_nodes;
};

struct Result {
    DistributedGraph        graph;
    scalable_vector<NodeID> mapping;
    MemoryContext           m_ctx;
};
} // namespace contraction

contraction::Result contract_local_clustering(
    const DistributedGraph& graph, const scalable_vector<Atomic<NodeID>>& clustering,
    contraction::MemoryContext m_ctx = {});
} // namespace dkaminpar::coarsening
