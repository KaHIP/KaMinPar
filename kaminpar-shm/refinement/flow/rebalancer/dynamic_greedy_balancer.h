#pragma once

#include <span>
#include <utility>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph> class SequentialDenseGainCache {
public:
  void
  initialize(const PartitionedGraph &p_graph, const Graph &graph, const BlockID overloaded_block) {
    _p_graph = &p_graph;
    _graph = &graph;
    _overloaded_block = overloaded_block;

    _n = graph.n();
    _k = p_graph.k();

    const std::size_t gain_cache_size = _n * _k;
    if (_gain_cache.size() < gain_cache_size) {
      _gain_cache.resize(gain_cache_size, static_array::noinit);
    }

    std::fill_n(_gain_cache.begin(), gain_cache_size, 0);
    for (const NodeID u : graph.nodes()) {
      if (_p_graph->block(u) != overloaded_block) {
        continue;
      }

      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _p_graph->block(v);
        _gain_cache[index(u, v_block)] += w;
      });
    }
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (_p_graph->block(v) != _overloaded_block) {
        return;
      }

      _gain_cache[index(v, from)] -= w;
      _gain_cache[index(v, to)] += w;
    });
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return connection(node, to) - connection(node, from);
  }

  [[nodiscard]] EdgeWeight connection(const NodeID node, const BlockID block) const {
    return _gain_cache[index(node, block)];
  }

private:
  [[nodiscard]] std::size_t index(const NodeID node, const BlockID block) const {
    return node * _k + block;
  }

private:
  const PartitionedGraph *_p_graph;
  const Graph *_graph;
  BlockID _overloaded_block;

  std::size_t _n;
  std::size_t _k;

  StaticArray<EdgeWeight> _gain_cache;
};

template <typename PartitionedGraph, typename Graph> class DynamicGreedyBalancer {
  using RelativeGain = float;
  using PriorityQueue = BinaryMaxHeap<RelativeGain>;
  using GainCache = SequentialDenseGainCache<PartitionedGraph, Graph>;

  struct Move {
    BlockID block;
    RelativeGain relative_gain;
  };

  [[nodiscard]] static RelativeGain
  compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) {
    return (absolute_gain >= 0) ? (absolute_gain * weight)
                                : (absolute_gain / static_cast<RelativeGain>(weight));
  }

public:
  DynamicGreedyBalancer(std::span<const BlockWeight> max_block_weights)
      : _max_block_weights(max_block_weights) {};

  std::pair<BlockID, RelativeGain>
  rebalance(PartitionedGraph &p_graph, const Graph &graph, BlockID overloaded_block) {
    _p_graph = &p_graph;
    _graph = &graph;

    _max_block_weights = p_graph.raw_block_weights();
    _overloaded_block = overloaded_block;

    _gain_cache.initialize(p_graph, graph, overloaded_block);

    insert_nodes();
    return move_nodes();
  }

private:
  void insert_nodes() {
    const NodeID num_nodes = _graph->n();
    if (_priority_queue.capacity() < num_nodes) {
      _priority_queue.resize(num_nodes);
    }
    if (_target_blocks.size() < num_nodes) {
      _target_blocks.resize(num_nodes, static_array::noinit);
    }

    _priority_queue.clear();
    for (const NodeID u : _graph->nodes()) {
      if (_p_graph->block(u) != _overloaded_block) {
        continue;
      }

      insert_node(u);
    }
  }

  std::pair<BlockID, RelativeGain> move_nodes() {
    EdgeWeight gain = 0;

    while (_p_graph->block_weight(_overloaded_block) > _max_block_weights[_overloaded_block]) {
      while (true) {
        if (_priority_queue.empty()) {
          return {false, 0};
        }

        const NodeID u = _priority_queue.peek_id();
        _priority_queue.pop();

        const BlockID target_block = _target_blocks[u];
        if (_p_graph->block_weight(target_block) + _graph->node_weight(u) >
            _max_block_weights[target_block]) {
          insert_node(u);
          continue;
        }

        gain += _gain_cache.gain(u, _overloaded_block, target_block);
        _gain_cache.move(u, _overloaded_block, target_block);

        _p_graph->set_block(u, target_block);
        _graph->adjacent_nodes(u, [&](const NodeID v) { update_node(v); });
        break;
      }
    }

    return {true, gain};
  }

  void insert_node(const NodeID u) {
    const auto [target_block, relative_gain] = compute_best_move(u);
    if (target_block == kInvalidBlockID) {
      return;
    }

    _priority_queue.push(u, relative_gain);
    _target_blocks[u] = target_block;
  }

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
    const BlockID u_block = _overloaded_block;
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

private:
  PartitionedGraph *_p_graph;
  const Graph *_graph;

  BlockID _overloaded_block;
  std::span<const BlockWeight> _max_block_weights;

  PriorityQueue _priority_queue;
  StaticArray<BlockID> _target_blocks;

  GainCache _gain_cache;
};

} // namespace kaminpar::shm
