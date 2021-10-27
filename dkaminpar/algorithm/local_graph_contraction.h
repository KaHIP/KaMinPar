/*******************************************************************************
 * @file:   local_graph_contraction.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "kaminpar/datastructure/ts_navigable_linked_list.h"

namespace dkaminpar::graph {
namespace contraction {
struct Edge {
  NodeID target;
  EdgeWeight weight;
};

struct MemoryContext {
  scalable_vector<NodeID> buckets;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>> buckets_index;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>> leader_mapping;
  scalable_vector<shm::NavigationMarker<NodeID, Edge>> all_buffered_nodes;
};

struct Result {
  DistributedGraph graph;
  scalable_vector<NodeID> mapping;
  MemoryContext m_ctx;
};
} // namespace contraction

contraction::Result contract_local_clustering(const DistributedGraph &graph, const scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>> &clustering,
                                              contraction::MemoryContext m_ctx = {});
} // namespace dkaminpar::graph