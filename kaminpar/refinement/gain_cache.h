/*******************************************************************************
 * @file:   gain_cache.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/partitioned_graph.h"

namespace kaminpar::shm {
class OnTheFlyGainCache {
public:
    void insert(const PartitionedGraph &p_graph, const NodeID u) {
    }

    void update(const PartitionedGraph &p_graph, const NodeID u, const BlockID from, const BlockID to) {

    }

    bool next(const PartitionedGraph &p_graph) {
        return false;
    }

    void clear() {}
};

class RecomputeGainCache {
public:
  void reinit(const PartitionedGraph *p_graph) { _p_graph = p_graph; }

  EdgeWeight gain_to(const NodeID u, const BlockID b) {
    EdgeWeight internal_degree = 0;
    EdgeWeight external_degree = 0;

    const BlockID u_block = _p_graph->block(u);
    for (const auto &[e, v] : _p_graph->neighbors(u)) {
      const BlockID v_block = _p_graph->block(v);
      const EdgeWeight e_weight = _p_graph->edge_weight(e);

      if (v_block == u_block) {
        internal_degree += e_weight;
      } else if (v_block == b) {
        external_degree += e_weight;
      }
    }

    return external_degree - internal_degree;
  }

  std::pair<BlockID, EdgeWeight> best_gain(const NodeID u,
                                           const BlockWeightsContext &ctx) {
    BlockID best_block = _p_graph->block(u);
    EdgeWeight best_gain = 0;

    const NodeWeight u_weight = _p_graph->node_weight(u);
    for (const BlockID b : _p_graph->blocks()) {
      if (_p_graph->block_weight(b) + u_weight > ctx.max(b)) {
        continue;
      }

      const EdgeWeight gain = gain_to(u, b);
      if (gain > best_gain) {
        best_block = b;
        best_gain = gain;
      }
    }
    return {best_block, best_gain};
  }

  bool is_border_node(const NodeID u) {
    const BlockID u_block = _p_graph->block(u);
    for (const auto &[e, v] : _p_graph->neighbors(u)) {
      const BlockID v_block = _p_graph->block(v);
      if (u_block != v_block) {
        return true;
      }
    }
    return false;
  }

  void move_node(const NodeID u, const BlockID from, const BlockID to) {
    ((void)u);
    ((void)from);
    ((void)to);
  }

private:
  const PartitionedGraph *_p_graph;
};
} // namespace kaminpar::shm
