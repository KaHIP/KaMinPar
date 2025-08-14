#pragma once

#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/rebalancer/greedy_balancer_base.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph>
class DynamicGreedyBalancer : GreedyBalancerBase<PartitionedGraph, Graph> {
  using Base = GreedyBalancerBase<PartitionedGraph, Graph>;

  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;
  using Base::_p_graph;

public:
  DynamicGreedyBalancer(std::span<const BlockWeight> max_block_weights)
      : Base(max_block_weights) {};

  void setup(PartitionedGraph &p_graph, const Graph &graph) {
    _p_graph = &p_graph;
    _graph = &graph;

    Base::initialize();
    _gain_cache.initialize(*_p_graph, *_graph);
  }

  void move_node(const NodeID u, const BlockID from, const BlockID to) {
    _gain_cache.move(u, from, to);
  }

  RebalancerResult rebalance(const BlockID overloaded_block) {
    _overloaded_block = overloaded_block;
    insert_nodes();

    _moved_nodes.clear();
    return move_nodes();
  }

private:
  void insert_nodes() {
    Base::clear_nodes();

    for (const NodeID u : _graph->nodes()) {
      if (_p_graph->block(u) == _overloaded_block) {
        Base::insert_node(u);
      }
    }
  }

  RebalancerResult move_nodes() {
    EdgeWeight gain = 0;

    const BlockWeight max_block_weight = _max_block_weights[_overloaded_block];
    while (_p_graph->block_weight(_overloaded_block) > max_block_weight) {
      while (true) {
        if (!Base::has_next_node()) {
          return RebalancerResult(false, 0, _moved_nodes);
        }

        const auto [u, target_block] = Base::next_node();
        if (_p_graph->block_weight(target_block) + _graph->node_weight(u) >
            _max_block_weights[target_block]) {
          Base::insert_node(u);
          continue;
        }

        gain += Base::move_node(u, _overloaded_block, target_block);
        _moved_nodes.push_back(u);

        break;
      }
    }

    return RebalancerResult(true, gain, _moved_nodes);
  }

private:
  BlockID _overloaded_block;
  ScalableVector<NodeID> _moved_nodes;
};

template <typename PartitionedGraph, typename Graph>
class DynamicGreedyMultiBalancer : GreedyBalancerBase<PartitionedGraph, Graph> {
  using Base = GreedyBalancerBase<PartitionedGraph, Graph>;

  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;
  using Base::_p_graph;

public:
  struct Move {
    NodeID node;
    BlockID from;
    BlockID to;
  };

  struct RebalancerResult {
    bool balanced;
    EdgeWeight gain;
    std::span<const Move> moves;
  };

public:
  DynamicGreedyMultiBalancer(std::span<const BlockWeight> max_block_weights)
      : Base(max_block_weights) {};

  RebalancerResult rebalance(PartitionedGraph &p_graph, const Graph &graph) {
    _p_graph = &p_graph;
    _graph = &graph;

    init_overloaded_blocks();
    if (_num_overloaded_blocks == 0) {
      return RebalancerResult(true, 0, {});
    }

    Base::initialize();
    _gain_cache.initialize(*_p_graph, *_graph);

    insert_nodes();
    const EdgeWeight gain = move_nodes();

    const bool balanced = _num_overloaded_blocks == 0;
    return RebalancerResult(balanced, gain, _moves);
  }

private:
  void init_overloaded_blocks() {
    const BlockID num_blocks = _p_graph->k();
    if (_is_overloaded.size() < num_blocks) {
      _is_overloaded.resize(num_blocks, static_array::noinit);
    }

    BlockID num_overloaded_blocks = 0;
    for (BlockID block = 0; block < num_blocks; ++block) {
      const bool is_overloaded = _p_graph->block_weight(block) > _max_block_weights[block];
      _is_overloaded[block] = is_overloaded;

      if (is_overloaded) {
        num_overloaded_blocks += 1;
      }
    }

    _num_overloaded_blocks = num_overloaded_blocks;
  };

  void insert_nodes() {
    Base::clear_nodes();

    for (const NodeID u : _graph->nodes()) {
      if (_is_overloaded[_p_graph->block(u)]) {
        Base::insert_node(u);
      }
    }
  }

  EdgeWeight move_nodes() {
    EdgeWeight gain = 0;

    _moves.clear();
    while (_num_overloaded_blocks > 0) {
      while (Base::has_next_node()) {
        const auto [u, target_block] = Base::next_node();

        const BlockID u_block = _p_graph->block(u);
        if (!_is_overloaded[u_block]) {
          continue;
        }

        if (_p_graph->block_weight(target_block) + _graph->node_weight(u) >
            _max_block_weights[target_block]) {
          Base::insert_node(u);
          continue;
        }

        gain += Base::move_node(u, u_block, target_block);
        _moves.emplace_back(u, u_block, target_block);

        if (_p_graph->block_weight(u_block) <= _max_block_weights[u_block]) {
          _num_overloaded_blocks -= 1;
          _is_overloaded[u_block] = false;
          break;
        }
      }
    };

    return gain;
  }

private:
  StaticArray<bool> _is_overloaded;
  BlockID _num_overloaded_blocks;

  ScalableVector<Move> _moves;
};

} // namespace kaminpar::shm
