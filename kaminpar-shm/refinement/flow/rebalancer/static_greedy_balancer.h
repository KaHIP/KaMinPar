#pragma once

#include <span>
#include <unordered_map>
#include <utility>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph> class StaticGreedyBalancer {
  using RelativeGain = float;
  using PriorityQueue = BinaryMaxHeap<RelativeGain>;

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
  StaticGreedyBalancer(std::span<const BlockWeight> max_block_weights)
      : _max_block_weights(max_block_weights) {};

  void initialize(
      PartitionedGraph &p_graph,
      const Graph &graph,
      const std::unordered_map<NodeID, NodeID> &global_to_local_mapping,
      BlockID overloaded_block
  ) {
    _p_graph = &p_graph;
    _graph = &graph;
    _global_to_local_mapping = &global_to_local_mapping;

    _max_block_weights = p_graph.raw_block_weights();
    _overloaded_block = overloaded_block;

    compute_moves();
  }

  std::pair<BlockID, RelativeGain> rebalance(PartitionedGraph &p_graph) {
    std::size_t cur_move = 0;
    while (p_graph.block_weight(_overloaded_block) > _max_block_weights[_overloaded_block]) {
      while (true) {
        if (cur_move >= _num_valid_moves) {
          return {false, 0};
        }

        const NodeID u = _moves[cur_move++];
        if (p_graph.block(u) != _overloaded_block) {
          continue;
        }

        const BlockID target_block = _target_blocks[u];
        if (p_graph.block_weight(target_block) + _graph->node_weight(u) >
            _max_block_weights[target_block]) {
          continue;
        }

        p_graph.set_block(u, target_block);
        break;
      }
    }

    return {true, kInvalidEdgeWeight}; // TODO: compute gain
  }

private:
  void compute_moves() {
    const NodeID num_nodes = _graph->n();
    if (_moves.size() < num_nodes) {
      _moves.resize(num_nodes, static_array::noinit);
    }
    if (_target_blocks.size() < num_nodes) {
      _target_blocks.resize(num_nodes, static_array::noinit);
    }
    if (_priority_queue.capacity() < num_nodes) {
      _priority_queue.resize(num_nodes);
    }

    const BlockID num_blocks = _p_graph->k();
    if (_local_connection.size() < num_blocks) {
      _local_connection.resize(num_blocks, static_array::noinit);
    }

    for (const NodeID u : _graph->nodes()) {
      if (_p_graph->block(u) != _overloaded_block && !_global_to_local_mapping->contains(u)) {
        continue;
      }

      insert_node(u);
    }

    _num_valid_moves = 0;
    while (!_priority_queue.empty()) {
      const NodeID u = _priority_queue.peek_id();
      _priority_queue.pop();

      _moves[_num_valid_moves++] = u;
    }
  }

  void insert_node(const NodeID u) {
    compute_local_connections(u);

    const auto [target_block, relative_gain] = compute_best_move(u);
    if (target_block == kInvalidBlockID) {
      return;
    }

    _priority_queue.push(u, relative_gain);
    _target_blocks[u] = target_block;
  }

  void compute_local_connections(const NodeID u) {
    std::fill_n(_local_connection.begin(), _p_graph->k(), 0);

    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      // TODO: should we assume that nodes of the other border region are also overloaded?
      const BlockID v_block =
          _global_to_local_mapping->contains(v) ? _overloaded_block : _p_graph->block(v);
      _local_connection[v_block] += w;
    });
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

      const EdgeWeight block_connection = _local_connection[block];
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

    const EdgeWeight from_connection = _local_connection[u_block];
    const EdgeWeight absolute_gain = target_block_connection - from_connection;

    const RelativeGain relative_gain = compute_relative_gain(absolute_gain, u_weight);
    return {target_block, relative_gain};
  }

private:
  PartitionedGraph *_p_graph;
  const Graph *_graph;
  const std::unordered_map<NodeID, NodeID> *_global_to_local_mapping;

  BlockID _overloaded_block;
  std::span<const BlockWeight> _max_block_weights;

  NodeID _num_valid_moves;
  StaticArray<NodeID> _moves;

  PriorityQueue _priority_queue;
  StaticArray<BlockID> _target_blocks;

  StaticArray<EdgeWeight> _local_connection;
};

} // namespace kaminpar::shm
