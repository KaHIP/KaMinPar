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
  using Base::_overloaded_block;
  using Base::_p_graph;

public:
  DynamicGreedyBalancer(std::span<const BlockWeight> max_block_weights)
      : Base(max_block_weights) {};

  void setup(PartitionedGraph &p_graph, const Graph &graph) {
    _p_graph = &p_graph;
    _graph = &graph;

    _initialized = false;
  }

  RebalancerResult rebalance(const BlockID overloaded_block) {
    _overloaded_block = overloaded_block;
    if (!_initialized) {
      initialize();
    }

    _gain_cache.initialize(_overloaded_block, *_p_graph, *_graph);
    insert_nodes();

    _moved_nodes.clear();
    return move_nodes();
  }

private:
  void initialize() {
    Base::initialize();
    _initialized = true;
  }

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

    while (_p_graph->block_weight(_overloaded_block) > _max_block_weights[_overloaded_block]) {
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

        gain += Base::move_node(u, target_block);
        _moved_nodes.push_back(u);

        break;
      }
    }

    return RebalancerResult(true, gain, _moved_nodes);
  }

private:
  bool _initialized;

  ScalableVector<NodeID> _moved_nodes;
};

} // namespace kaminpar::shm
