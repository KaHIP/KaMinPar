/*******************************************************************************
 * Contracts clusterings and constructs the coarse graph.
 *
 * @file:   cluster_contraction.h
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/ts_navigable_linked_list.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm {
class CoarseGraph {
public:
  virtual ~CoarseGraph() = default;

  virtual const Graph &get() const = 0;
  virtual Graph &get() = 0;

  virtual void project(const StaticArray<BlockID> &array, StaticArray<BlockID> &onto) = 0;
};

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
} // namespace contraction

std::unique_ptr<CoarseGraph> contract(
    const Graph &graph,
    const ContractionCoarseningContext &con_ctx,
    scalable_vector<parallel::Atomic<NodeID>> &clustering
);

std::unique_ptr<CoarseGraph> contract(
    const Graph &graph,
    const ContractionCoarseningContext &con_ctx,
    scalable_vector<parallel::Atomic<NodeID>> &clustering,
    contraction::MemoryContext &m_ctx
);
} // namespace kaminpar::shm
