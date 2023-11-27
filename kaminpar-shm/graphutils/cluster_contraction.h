/*******************************************************************************
 * Contracts clusterings and constructs the coarse graph.
 *
 * @file:   cluster_contraction.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm::graph {
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
  Graph graph;
  scalable_vector<NodeID> mapping;
  MemoryContext m_ctx;
};
} // namespace contraction

contraction::Result contract(
    const Graph &r, const scalable_vector<NodeID> &clustering, contraction::MemoryContext m_ctx = {}
);

contraction::Result contract(
    const Graph &graph,
    const scalable_vector<parallel::Atomic<NodeID>> &clustering,
    contraction::MemoryContext m_ctx = {}
);
} // namespace kaminpar::shm::graph
