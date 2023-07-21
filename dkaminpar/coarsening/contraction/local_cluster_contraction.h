/*******************************************************************************
 * Graph contraction for local clusterings.
 *
 * @file:   local_cluster_contraction.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"

#include "common/datastructures/ts_navigable_linked_list.h"
#include "common/parallel/atomic.h"
#include "common/datastructures/scalable_vector.h"

namespace kaminpar::dist {
namespace contraction {
struct Edge {
  NodeID target;
  EdgeWeight weight;
};

struct MemoryContext {
  scalable_vector<NodeID> buckets;
  scalable_vector<parallel::Atomic<NodeID>> buckets_index;
  scalable_vector<parallel::Atomic<NodeID>> leader_mapping;
  scalable_vector<NavigationMarker<NodeID, Edge, scalable_vector>> all_buffered_nodes;
};

struct Result {
  DistributedGraph graph;
  scalable_vector<NodeID> mapping;
  MemoryContext m_ctx;
};
} // namespace contraction

contraction::Result contract_local_clustering(
    const DistributedGraph &graph,
    const scalable_vector<parallel::Atomic<NodeID>> &clustering,
    contraction::MemoryContext m_ctx = {}
);
} // namespace kaminpar::dist
