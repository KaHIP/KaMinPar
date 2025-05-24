#pragma once

#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class PiercingHeuristic {
  SET_DEBUG(false);

public:
  PiercingHeuristic(
      const PiercingHeuristicContext &_ctx,
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &initial_source_side_nodes,
      const std::unordered_set<NodeID> &initial_sink_side_nodes
  );

  ScalableVector<NodeID> pierce_on_source_side(
      const NodeStatus &cut_status, const NodeStatus &terminal_status, NodeWeight max_weight
  );

  ScalableVector<NodeID> pierce_on_sink_side(
      const NodeStatus &cut_status, const NodeStatus &terminal_status, NodeWeight max_weight
  );

private:
  ScalableVector<NodeID> find_piercing_node(
      const NodeStatus &cut_status,
      const NodeStatus &terminal_status,
      const std::unordered_set<NodeID> &initial_terminal_side_nodes,
      const NodeWeight max_weight,
      const bool source_side
  );

  void compute_distances();

  [[nodiscard]] StaticArray<NodeID> compute_distances(NodeID terminal);

private:
  const PiercingHeuristicContext &_ctx;

  const CSRGraph &_graph;
  const std::unordered_set<NodeID> &_initial_source_side_nodes;
  const std::unordered_set<NodeID> &_initial_sink_side_nodes;

  StaticArray<NodeID> _distance;
};

} // namespace kaminpar::shm
