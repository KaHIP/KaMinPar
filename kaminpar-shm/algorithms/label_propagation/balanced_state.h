/*******************************************************************************
 * Partition state used by balanced label-propagation refinement.
 *
 * @file:   balanced_state.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <span>

#include "kaminpar-shm/algorithms/label_propagation/active_set.h"
#include "kaminpar-shm/algorithms/label_propagation/move.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::shm::lp {

class BalancedState {
public:
  void reset(
      PartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const std::span<const NodeID> communities
  ) {
    _p_graph = &p_graph;
    _p_ctx = &p_ctx;
    _communities = communities;
    _active.reset(p_graph.n());
  }

  [[nodiscard]] BlockID cluster(const NodeID u) const {
    return _p_graph->block(u);
  }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID block) const {
    return _p_graph->block_weight(block);
  }

  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID block) const {
    return _p_ctx->max_block_weight(block);
  }

  [[nodiscard]] BlockWeight min_cluster_weight(const BlockID block) const {
    return _p_ctx->min_block_weight(block);
  }

  [[nodiscard]] bool accepts_neighbor(const NodeID u, const NodeID v) const {
    return _communities.empty() || _communities[u] == _communities[v];
  }

  template <typename Graph>
  MoveResult commit(
      const Graph &graph,
      const NodeID u,
      const BlockID from,
      const BlockID to,
      const NodeWeight u_weight
  ) {
    if (cluster(u) == to || !_p_graph->move_block_weight(
                                from, to, u_weight, max_cluster_weight(to), min_cluster_weight(from)
                            )) {
      return {};
    }

    _p_graph->set_block<false>(u, to);
    _active.activate_neighbors(graph, u, [](const NodeID) { return true; });
    return {.moved = true, .emptied_cluster = false};
  }

  [[nodiscard]] bool is_active(const NodeID u) const {
    return _active.contains(u);
  }

  void deactivate(const NodeID u) {
    _active.deactivate(u);
  }

  void free() {
    _active.free();
    _p_graph = nullptr;
    _p_ctx = nullptr;
  }

private:
  PartitionedGraph *_p_graph = nullptr;
  const PartitionContext *_p_ctx = nullptr;
  std::span<const NodeID> _communities;
  ActiveSet _active;
};

} // namespace kaminpar::shm::lp
