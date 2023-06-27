#include "dkaminpar/refinement/balancer/move_sets.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "dkaminpar/context.h"

#include "common/assertion_levels.h"
#include "common/datastructures/binary_heap.h"
#include "common/datastructures/fast_reset_array.h"
#include "common/datastructures/marker.h"
#include "common/datastructures/rating_map.h"
#include "common/noinit_vector.h"

namespace kaminpar::dist {
MoveSets::MoveSets(
    const DistributedPartitionedGraph &p_graph,
    NoinitVector<NodeID> node_to_move_set,
    NoinitVector<NodeID> move_sets,
    NoinitVector<NodeID> move_set_indices
)
    : _p_graph(p_graph),
      _node_to_move_set(std::move(node_to_move_set)),
      _move_sets(std::move(move_sets)),
      _move_set_indices(std::move(move_set_indices)) {
  KASSERT(_move_set_indices.front() == 0u);
  KASSERT(_move_set_indices.back() == _move_sets.size());
}

NodeID MoveSets::size(const NodeID set) const {
  KASSERT(set + 1 < _move_set_indices.size());
  return _move_set_indices[set + 1] - _move_set_indices[set];
}

namespace {
class MoveSetBuilder {
public:
  MoveSetBuilder(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx)
      : _p_graph(p_graph),
        _p_ctx(p_ctx),
        _node_to_move_set(p_graph.n()),
        _move_sets(p_graph.n()),
        _move_set_indices(p_graph.n() + 1),
        _conns(p_graph.n() * p_graph.k()),
        _cur_conns(p_graph.k()) {
    _p_graph.pfor_nodes([&](const NodeID u) {
      _node_to_move_set[u] = kInvalidNodeID;
      _move_sets[u] = kInvalidNodeID;
    });
    _move_set_indices.front() = 0;
  }

  void grow_move_set(const NodeID u) {
    KASSERT(_cur_block == kInvalidBlockID || _cur_block == _p_graph.block(u));

    if (_cur_block == kInvalidBlockID) {
      _cur_block = _p_graph.block(u);
    }

    _cur_weight += _p_graph.node_weight(u);
    _node_to_move_set[u] = _cur_move_set;
    _move_sets[_cur_pos] = u;
    ++_cur_pos;

    for (const auto [e, v] : _p_graph.neighbors(u)) {
      if (_p_graph.is_owned_node(v) && _node_to_move_set[v] == _cur_move_set) {
        _cur_block_conn -= _p_graph.edge_weight(e);
      } else {
        const BlockID bv = _p_graph.block(v);
        if (bv == _cur_block) {
          _cur_block_conn += _p_graph.edge_weight(e);
        } else {
          _cur_conns.decrease_priority_by(bv, _p_graph.edge_weight(e));
        }
      }
    }
  }

  void finish_move_set() {
    reset_cur_conns();
    _cur_block = kInvalidBlockID;
    _cur_block_conn = 0;

    _move_set_indices[_cur_move_set + 1] = _cur_pos;
    ++_cur_move_set;
  }

  MoveSets finalize() {
    _move_set_indices.resize(_cur_move_set + 1);
    _move_set_indices.back() = _p_graph.n();

    return {
        _p_graph,
        std::move(_node_to_move_set),
        std::move(_move_sets),
        std::move(_move_set_indices),
    };
  }

private:
  void reset_cur_conns() {
    _cur_conns.clear();
    for (const BlockID b : _p_graph.blocks()) {
      _cur_conns.push(b, 0);
    }
  }

  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  NoinitVector<NodeID> _node_to_move_set;
  NoinitVector<NodeID> _move_sets;
  NoinitVector<NodeID> _move_set_indices;

  NoinitVector<EdgeWeight> _conns;

  NodeID _cur_pos = 0;
  NodeID _cur_move_set = 0;
  EdgeWeight _cur_block_conn = 0;
  BinaryMaxHeap<EdgeWeight> _cur_conns;
  BlockID _cur_block = kInvalidBlockID;
  NodeWeight _cur_weight = 0;

  NodeID _best_prefix_pos = 0;
  BlockID _best_prefix_block = kInvalidBlockID;
  EdgeWeight _best_prefix_gain = 0;
};
} // namespace

MoveSets build_greedy_move_sets(
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    const NodeWeight max_move_set_weight
) {
  MoveSetBuilder builder(p_graph, p_ctx);

  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);
    if (p_graph.block_weight(bu) <= p_ctx.graph->max_block_weight(bu)) {
      continue;
    }
  }

  NoinitVector<NodeID> move_sets(p_graph.n());
  std::vector<NodeID> move_set_indices;
  std::vector<EdgeWeight> move_set_gains(p_graph.n() * p_graph.k());
  Marker marker(p_graph.n());

  NodeWeight current_move_set_weight = 0;
  std::size_t current_position = 0;
  std::size_t best_position = 0;
  move_set_indices.push_back(0);

  BinaryMinHeap<EdgeWeight> gains(p_graph.k());
  BinaryMaxHeap<EdgeWeight> frontier(p_graph.n());

  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);

    // Only consider nodes in overloaded blocks
    if (p_graph.block_weight(bu) <= p_ctx.graph->max_block_weight(bu)) {
      continue;
    }

    // Don't consider nodes twice
    if (marker.get(u)) {
      continue;
    }

    move_sets[current_position] = u;
    current_move_set_weight += p_graph.node_weight(u);
    marker.set(u);

    for (const auto [e, v] : p_graph.neighbors(u)) {
      const BlockID bv = p_graph.block(v);

      if (gains.contains(bv)) {
        gains.decrease_priority_by(bv, p_graph.edge_weight(e));
      } else {
        gains.push(bv, p_graph.edge_weight(e));
      }

      if (p_graph.is_owned_node(v) && bv == bu && !marker.get(v)) {
        if (frontier.contains(v)) {
          frontier.decrease_priority_by(v, p_graph.edge_weight(e));
        } else {
          frontier.push(v, p_graph.edge_weight(e));
        }
      }
    }

    if (current_move_set_weight >= max_move_set_weight) {
    }
  }

  return builder.finalize();
}
} // namespace kaminpar::dist

