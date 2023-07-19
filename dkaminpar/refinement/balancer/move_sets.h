#pragma once

#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

#include "common/datastructures/fast_reset_array.h"
#include "common/ranges.h"

namespace kaminpar::dist {
struct MoveSetsMemoryContext {
  NoinitVector<NodeID> node_to_move_set;
  NoinitVector<NodeID> move_sets;
  NoinitVector<NodeID> move_set_indices;
  NoinitVector<EdgeWeight> move_set_conns;
};

class MoveSets {
public:
  MoveSets(const DistributedPartitionedGraph &p_graph, MoveSetsMemoryContext m_ctx);

  MoveSets(
      const DistributedPartitionedGraph &p_graph,
      NoinitVector<NodeID> node_to_move_set,
      NoinitVector<NodeID> move_sets,
      NoinitVector<NodeID> move_set_indices,
      NoinitVector<EdgeWeight> move_set_conns
  );

  operator MoveSetsMemoryContext() &&;

  [[nodiscard]] inline NodeID size(const NodeID set) const {
    KASSERT(set + 1 < _move_set_indices.size());
    return _move_set_indices[set + 1] - _move_set_indices[set];
  }

  [[nodiscard]] inline auto sets() const {
    return IotaRange<NodeID>(0, num_move_sets());
  }

  [[nodiscard]] inline auto elements(const NodeID set) const {
    return TransformedIotaRange(
        _move_set_indices[set],
        _move_set_indices[set + 1],
        [this](const NodeID i) { return _move_sets[i]; }
    );
  }

  [[nodiscard]] inline EdgeWeight gain(const NodeID set, const BlockID to_block) const {
    return conn(set, to_block) - conn(set, block(set));
  }

  [[nodiscard]] inline double relative_gain(const NodeID set, const BlockID to_block) const {
    return compute_relative_gain(gain(set, to_block), weight(set));
  }

  [[nodiscard]] inline EdgeWeight conn(const NodeID set, const BlockID to_block) const {
    return _move_set_conns[set * _p_graph.k() + to_block];
  }

  [[nodiscard]] inline BlockID block(const NodeID set) const {
    return _p_graph.block(_move_sets[_move_set_indices[set]]);
  }

  [[nodiscard]] inline NodeWeight weight(const NodeID set) const {
    NodeWeight weight = 0;
    for (const NodeID u : elements(set)) {
      weight += _p_graph.node_weight(u);
    }
    return weight;
  }

  [[nodiscard]] inline NodeID num_move_sets() const {
    return _move_set_indices.size() - 1;
  }

  inline void move_ghost_node(const NodeID ghost, const BlockID from, const BlockID to) {
    KASSERT(_p_graph.is_ghost_node(ghost));
    const NodeID nth_ghost = ghost - _p_graph.ghost_n();

    for (EdgeID edge = _ghost_node_indices[nth_ghost]; edge < _ghost_node_indices[nth_ghost + 1];
         ++edge) {
      const auto [weight, set] = _ghost_node_edges[edge];
      _move_set_conns[set * _p_graph.k() + from] -= weight;
      _move_set_conns[set * _p_graph.k() + to] += weight;
    }
  }

  inline NodeID set_of(const NodeID node) const {
    return _node_to_move_set[node];
  }

  inline void move_set(const NodeID set, const BlockID from, const BlockID to) {
    for (const NodeID u : elements(set)) {
      for (const auto [e, v] : _p_graph.neighbors(u)) {
        if (!_p_graph.contains_local_node(v)) {
          continue;
        }

        const NodeID set_v = _node_to_move_set[v];
        if (set_v == kInvalidNodeID || set_v == set) {
          continue;
        }

        const EdgeWeight delta = _p_graph.edge_weight(e);
        _move_set_conns[set_v * _p_graph.k() + from] -= delta;
        _move_set_conns[set_v * _p_graph.k() + to] += delta;
      }
    }
  }

  [[nodiscard]] inline BlockID owner(const NodeID set) const {
    return block(_move_sets[_move_set_indices[set]]);
  }

  inline std::pair<EdgeWeight, BlockID> find_max_conn(const NodeID set) const {
    KASSERT(size(set) > 0);

    EdgeWeight max_conn = std::numeric_limits<EdgeWeight>::min();
    BlockID max_gainer = kInvalidBlockID;

    const BlockID set_b = owner(set);
    for (const BlockID b : _p_graph.blocks()) {
      if (b != set_b && conn(set, b) > max_conn) {
        max_conn = conn(set, b);
        max_gainer = b;
      }
    }

    KASSERT(max_conn >= 0);
    KASSERT(max_gainer != kInvalidBlockID);

    return {max_conn, max_gainer};
  }

  inline std::pair<EdgeWeight, BlockID> find_max_gain(const NodeID set) const {
    const auto [max_conn, max_gainer] = find_max_conn(set);
    return {max_conn - conn(set, owner(set)), max_gainer};
  }

  inline std::pair<double, BlockID> find_max_relative_gain(const NodeID set) const {
    const auto [absolute_gain, max_gainer] = find_max_gain(set);
    return {compute_relative_gain(absolute_gain, weight(set)), max_gainer};
  }

private:
  double compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight set_weight) const {
    if (absolute_gain >= 0) {
      return absolute_gain * set_weight;
    } else {
      return 1.0 * absolute_gain / set_weight;
    }
  }

  void init_ghost_node_adjacency();

  const DistributedPartitionedGraph &_p_graph;

  NoinitVector<NodeID> _node_to_move_set;
  NoinitVector<NodeID> _move_sets;
  NoinitVector<NodeID> _move_set_indices;
  NoinitVector<EdgeWeight> _move_set_conns;

  NoinitVector<NodeID> _ghost_node_indices;
  NoinitVector<std::pair<NodeID, EdgeWeight>> _ghost_node_edges;
};

MoveSets build_greedy_move_sets(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    NodeWeight max_move_set_size,
    MoveSetsMemoryContext m_ctx
);
} // namespace kaminpar::dist
