#pragma once

#include <span>
#include <unordered_map>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/rebalancer/gain_cache.h"
#include "kaminpar-shm/refinement/flow/rebalancer/greedy_balancer_base.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph>
class StaticGreedyBalancer : GreedyBalancerBase<
                                 PartitionedGraph,
                                 Graph,
                                 PinnedNonConcurrentDenseGainCache<
                                     PartitionedGraph,
                                     Graph,
                                     std::unordered_map<NodeID, NodeID>>> {
  using Base = GreedyBalancerBase<
      PartitionedGraph,
      Graph,
      PinnedNonConcurrentDenseGainCache<
          PartitionedGraph,
          Graph,
          std::unordered_map<NodeID, NodeID>>>;

  using Base::_gain_cache;
  using Base::_graph;
  using Base::_max_block_weights;
  using Base::_overloaded_block;
  using Base::_p_graph;

  struct Move {
    NodeID node;
    BlockID target_block;
  };

public:
  StaticGreedyBalancer(std::span<const BlockWeight> max_block_weights) : Base(max_block_weights) {};

  void initialize(
      BlockID overloaded_block,
      PartitionedGraph &p_graph,
      const Graph &graph,
      const std::unordered_map<NodeID, NodeID> &global_to_local_mapping
  ) {
    Base::initialize(overloaded_block, p_graph, graph);

    _gain_cache.initialize(overloaded_block, global_to_local_mapping, p_graph, graph);
    compute_moves(global_to_local_mapping);
  }

  RebalancerResult rebalance(PartitionedGraph &p_graph) {
    _moved_nodes.clear();

    const NodeID num_moves = _moves.size();
    NodeID cur_move = 0;
    while (p_graph.block_weight(_overloaded_block) > _max_block_weights[_overloaded_block]) {
      while (true) {
        if (cur_move >= num_moves) {
          return RebalancerResult(false, 0, _moved_nodes);
        }

        const auto [u, target_block] = _moves[cur_move++];
        if (p_graph.block(u) != _overloaded_block) {
          continue;
        }

        if (p_graph.block_weight(target_block) + _graph->node_weight(u) >
            _max_block_weights[target_block]) {
          continue;
        }

        p_graph.set_block(u, target_block);
        _moved_nodes.push_back(u);
        break;
      }
    }

    return RebalancerResult(true, kInvalidEdgeWeight, _moved_nodes); // TODO: compute gain
  }

private:
  void compute_moves(const std::unordered_map<NodeID, NodeID> &global_to_local_mapping) {
    _moves.clear();
    _moved_nodes.clear();

    for (const NodeID u : _graph->nodes()) {
      if (_p_graph->block(u) == _overloaded_block || global_to_local_mapping.contains(u)) {
        Base::insert_node(u);
      }
    }

    while (Base::has_next_node()) {
      const auto [u, target_block] = Base::next_node();
      _moves.emplace_back(u, target_block);
    }
  }

private:
  ScalableVector<Move> _moves;
  ScalableVector<NodeID> _moved_nodes;
};

} // namespace kaminpar::shm
