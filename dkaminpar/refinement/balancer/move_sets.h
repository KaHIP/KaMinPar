#pragma once

#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

#include "common/ranges.h"

namespace kaminpar::dist {
class MoveSets {
public:
  MoveSets(
      const DistributedPartitionedGraph &p_graph,
      NoinitVector<NodeID> node_to_move_set,
      NoinitVector<NodeID> move_sets,
      NoinitVector<NodeID> move_set_indices
  );

  [[nodiscard]] NodeID size(const NodeID set) const;

  [[nodiscard]] inline auto set(const NodeID set) const {
    return TransformedIotaRange(
        _move_set_indices[set],
        _move_set_indices[set + 1],
        [this](const NodeID i) { return _move_sets[i]; }
    );
  }

  [[nodiscard]] inline EdgeWeight gain(const NodeID set, const BlockID to_block) const {
    return conn(set, to_block) - conn(set, block(set));
  }

  [[nodiscard]] inline EdgeWeight conn(const NodeID set, const BlockID to_block) const {
    return _move_set_conns[set * _p_graph.k() + to_block];
  }

  [[nodiscard]] inline BlockID block(const NodeID set) const {
    return _p_graph.block(_move_sets[_move_set_indices[set]]);
  }

  NodeID num_move_sets() const {
    return _move_set_indices.size() - 1;
  }

private:
  const DistributedPartitionedGraph &_p_graph;

  NoinitVector<NodeID> _node_to_move_set;
  NoinitVector<NodeID> _move_sets;
  NoinitVector<NodeID> _move_set_indices;
  NoinitVector<EdgeWeight> _move_set_conns;
};

MoveSets build_greedy_move_sets(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    NodeWeight max_move_set_size
);
} // namespace kaminpar::dist
