#pragma once

#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/rebalancer/greedy_balancer_base.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph>
class StaticGreedyBalancer : GreedyBalancerBase<PartitionedGraph, Graph> {
  using Base = GreedyBalancerBase<PartitionedGraph, Graph>;

  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;
  using Base::_p_graph;

  struct Move {
    NodeID node;
    BlockID target_block;
  };

public:
  StaticGreedyBalancer(std::span<const BlockWeight> max_block_weights) : Base(max_block_weights) {};

  void setup(
      BlockID overloaded_block,
      PartitionedGraph &p_graph,
      const Graph &graph,
      const DynamicRememberingFlatMap<NodeID, NodeID> &global_to_local_mapping
  ) {
    _overloaded_block = overloaded_block;
    _p_graph = &p_graph;
    _graph = &graph;

    _initialized = false;
    _global_to_local_mapping = &global_to_local_mapping;
  }

  RebalancerResult rebalance() {
    if (!_initialized) {
      initialize();
    }

    _moved_nodes.clear();
    return move_nodes();
  }

private:
  void initialize() {
    SCOPED_TIMER("Initialize");

    Base::initialize();
    _initialized = true;

    _virtual_moves.clear();
    for (const auto &[u, _] : _global_to_local_mapping->entries()) {
      const BlockID u_block = _p_graph->block(u);
      if (u_block == _overloaded_block) {
        continue;
      }

      _virtual_moves.emplace_back(u, u_block);
      _p_graph->set_block(u, _overloaded_block);
    };

    _gain_cache.initialize(*_p_graph, *_graph);
    compute_moves();

    for (const NodeID u : _moved_nodes) {
      _p_graph->set_block(u, _overloaded_block);
    }
    for (const auto &[u, u_block] : _virtual_moves) {
      _p_graph->set_block(u, u_block);
    }
  }

  void compute_moves() {
    _moves.clear();
    _moved_nodes.clear();

    Base::clear_nodes();
    for (const NodeID u : _graph->nodes()) {
      if (_p_graph->block(u) == _overloaded_block) {
        Base::insert_node(u);
      }
    }

    while (Base::has_next_node()) {
      const auto [u, target_block] = Base::next_node();

      if (_p_graph->block_weight(target_block) + _graph->node_weight(u) >
          _max_block_weights[target_block]) {
        Base::insert_node(u);
        continue;
      }

      Base::move_node(u, _overloaded_block, target_block);
      _moves.emplace_back(u, target_block);

      _moved_nodes.push_back(u);
    }
  }

  RebalancerResult move_nodes() {
    SCOPED_TIMER("Compute moves");

    const NodeID num_moves = _moves.size();
    NodeID cur_move = 0;

    EdgeWeight gain = 0;
    while (_p_graph->block_weight(_overloaded_block) > _max_block_weights[_overloaded_block]) {
      while (true) {
        if (cur_move >= num_moves) {
          return RebalancerResult(false, 0, _moved_nodes);
        }

        const auto [u, target_block] = _moves[cur_move++];
        if (_p_graph->block(u) != _overloaded_block) {
          continue;
        }

        if (_p_graph->block_weight(target_block) + _graph->node_weight(u) >
            _max_block_weights[target_block]) {
          continue;
        }

        _p_graph->set_block(u, target_block);
        _moved_nodes.push_back(u);

        EdgeWeight from_connection = 0;
        EdgeWeight to_connection = 0;
        _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          const BlockID v_block = _p_graph->block(v);
          from_connection += (v_block == _overloaded_block) ? w : 0;
          to_connection += (v_block == target_block) ? w : 0;
        });

        gain += to_connection - from_connection;
        break;
      }
    }

    return RebalancerResult(true, gain, _moved_nodes);
  }

private:
  bool _initialized;
  const DynamicRememberingFlatMap<NodeID, NodeID> *_global_to_local_mapping;
  ScalableVector<Move> _virtual_moves;

  BlockID _overloaded_block;
  ScalableVector<Move> _moves;
  ScalableVector<NodeID> _moved_nodes;
};

} // namespace kaminpar::shm
