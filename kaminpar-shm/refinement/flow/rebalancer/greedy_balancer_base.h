#pragma once

#include <span>
#include <utility>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/rebalancer/gain_cache.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

struct RebalancerResult {
  bool balanced;
  EdgeWeight gain;
  std::span<const NodeID> moved_nodes;
};

template <typename PartitionedGraph, typename Graph> class GreedyBalancerBase {
  using RelativeGain = float;
  using PriorityQueue = BinaryMaxHeap<RelativeGain>;
  using GainCache = NonConcurrentDenseGainCache<PartitionedGraph, Graph>;

  struct Move {
    BlockID block;
    RelativeGain relative_gain;
  };

protected:
  GreedyBalancerBase(std::span<const BlockWeight> max_block_weights)
      : _max_block_weights(max_block_weights) {};

  void initialize() {
    if (_priority_queue.capacity() < _graph->n()) {
      _priority_queue.resize(_graph->n());
    }

    if (_target_blocks.size() < _graph->n()) {
      _target_blocks.resize(_graph->n(), static_array::noinit);
    }
  }

  [[nodiscard]] bool has_next_node() const {
    return !_priority_queue.empty();
  }

  std::pair<NodeID, BlockID> next_node() {
    const NodeID u = _priority_queue.peek_id();
    _priority_queue.pop();

    const BlockID target_block = _target_blocks[u];
    return {u, target_block};
  }

  void clear_nodes() {
    _priority_queue.clear();
  }

  void insert_node(const NodeID u) {
    const auto [target_block, relative_gain] = compute_best_move(u);
    if (target_block == kInvalidBlockID) {
      return;
    }

    _priority_queue.push(u, relative_gain);
    _target_blocks[u] = target_block;
  }

  EdgeWeight move_node(const NodeID u, const BlockID source_block, const BlockID target_block) {
    const EdgeWeight gain = _gain_cache.gain(u, source_block, target_block);
    _gain_cache.move(u, source_block, target_block);

    _p_graph->set_block(u, target_block);
    _graph->adjacent_nodes(u, [&](const NodeID v) { update_node(v); });

    return gain;
  }

private:
  void update_node(const NodeID u) {
    if (!_priority_queue.contains(u)) {
      return;
    }

    const auto [target_block, relative_gain] = compute_best_move(u);
    if (target_block == kInvalidBlockID) {
      _priority_queue.remove(u);
      return;
    }

    _priority_queue.change_priority(u, relative_gain);
    _target_blocks[u] = target_block;
  }

  [[nodiscard]] Move compute_best_move(const NodeID u) const {
    const BlockID u_block = _p_graph->block(u);
    const NodeWeight u_weight = _graph->node_weight(u);

    BlockID target_block = kInvalidBlockID;
    BlockWeight target_block_weight = std::numeric_limits<BlockWeight>::max();
    EdgeWeight target_block_connection = std::numeric_limits<RelativeGain>::min();

    const BlockID num_blocks = _p_graph->k();
    for (BlockID block = 0; block < num_blocks; ++block) {
      if (block == u_block) {
        continue;
      }

      const BlockWeight block_weight = _p_graph->block_weight(block);
      if (block_weight + u_weight > _max_block_weights[block]) {
        continue;
      }

      const EdgeWeight block_connection = _gain_cache.connection(u, block);
      if (block_connection > target_block_connection ||
          (block_connection == target_block_connection && block_weight < target_block_weight)) {
        target_block = block;
        target_block_weight = block_weight;
        target_block_connection = block_connection;
      }
    }

    if (target_block == kInvalidBlockID) {
      return {kInvalidBlockID, 0};
    }

    const EdgeWeight from_connection = _gain_cache.connection(u, u_block);
    const EdgeWeight absolute_gain = target_block_connection - from_connection;

    const RelativeGain relative_gain = compute_relative_gain(absolute_gain, u_weight);
    return {target_block, relative_gain};
  }

  [[nodiscard]] static RelativeGain
  compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) {
    return (absolute_gain >= 0) ? (absolute_gain * weight)
                                : (absolute_gain / static_cast<RelativeGain>(weight));
  }

protected:
  std::span<const BlockWeight> _max_block_weights;

  PartitionedGraph *_p_graph;
  const Graph *_graph;

  GainCache _gain_cache;

private:
  PriorityQueue _priority_queue;
  StaticArray<BlockID> _target_blocks;
};

} // namespace kaminpar::shm
