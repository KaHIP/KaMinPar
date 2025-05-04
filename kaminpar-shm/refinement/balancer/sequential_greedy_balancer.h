/*******************************************************************************
 * Sequential greedy balancing algorithm.
 *
 * @file:   sequential_greedy_balancer.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <utility>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

namespace {

template <typename PartitionedGraph, typename Graph> class SequentialDenseGainCache {
public:
  void initialize(const PartitionedGraph &p_graph, const Graph &graph) {
    _p_graph = &p_graph;
    _graph = &graph;

    _n = graph.n();
    _k = p_graph.k();

    const std::size_t gain_cache_size = _n * _k;
    if (_gain_cache.size() < gain_cache_size) {
      _gain_cache.resize(gain_cache_size, static_array::noinit);
    }

    std::fill_n(_gain_cache.begin(), gain_cache_size, 0);
    for (NodeID u = 0; u < _n; ++u) {
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _p_graph->block(v);
        _gain_cache[index(u, v_block)] += w;
      });
    }
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
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

  std::size_t _n;
  std::size_t _k;

  StaticArray<EdgeWeight> _gain_cache;
};

} // namespace

template <typename PartitionedGraph, typename Graph> class SequentialGreedyBalancerImpl {
  using RelativeGain = float;
  using PriorityQueue = BinaryMaxHeap<RelativeGain>;
  using GainCache = SequentialDenseGainCache<PartitionedGraph, Graph>;

  [[nodiscard]] static RelativeGain
  compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) {
    return (absolute_gain >= 0) ? (absolute_gain * weight)
                                : (absolute_gain / static_cast<RelativeGain>(weight));
  }

public:
  struct Result {
    bool rebalanced;
    EdgeWeight gain;
  };

  SequentialGreedyBalancerImpl() : _priority_queue(0) {}

  Result balance(
      PartitionedGraph &p_graph, const Graph &graph, std::span<const BlockWeight> max_block_weights
  ) {
    _p_graph = &p_graph;
    _graph = &graph;
    _max_block_weights = max_block_weights;

    init_overloaded_blocks();
    if (_num_overloaded_blocks == 0) {
      return Result(false, 0);
    }

    _gain = 0;
    _gain_cache.initialize(p_graph, graph);

    insert_nodes();
    move_nodes();

    const bool balanced = _num_overloaded_blocks == 0;
    return Result(balanced, _gain);
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
      num_overloaded_blocks += is_overloaded ? 1 : 0;
    }

    _num_overloaded_blocks = num_overloaded_blocks;
  };

  void insert_nodes() {
    const NodeID num_nodes = _graph->n();
    if (_target_blocks.size() < num_nodes) {
      _target_blocks.resize(num_nodes, static_array::noinit);
    }

    _priority_queue = PriorityQueue(num_nodes);
    for (NodeID u = 0; u < num_nodes; ++u) {
      const BlockID u_block = _p_graph->block(u);
      if (!_is_overloaded[u_block]) {
        continue;
      }

      insert_node(u, u_block);
    }
  }

  void move_nodes() {
    while (_num_overloaded_blocks > 0 && !_priority_queue.empty()) {
      const NodeID u = _priority_queue.peek_id();
      _priority_queue.pop();

      const BlockID u_block = _p_graph->block(u);
      if (!_is_overloaded[u_block]) {
        continue;
      }

      const BlockID target_block = _target_blocks[u];
      move_node(u, u_block, target_block);
    };
  }

  void move_node(const NodeID u, const BlockID from, const BlockID to) {
    if (_p_graph->block_weight(to) + _graph->node_weight(u) > _max_block_weights[to]) {
      insert_node(u, from);
      return;
    }

    _gain += _gain_cache.gain(u, from, to);
    _gain_cache.move(u, from, to);

    _p_graph->set_block(u, to);
    _graph->adjacent_nodes(u, [&](const NodeID v) { update_node(v); });

    if (_p_graph->block_weight(from) <= _max_block_weights[from]) {
      _num_overloaded_blocks -= 1;
      _is_overloaded[from] = false;
    }
  }

  void insert_node(const NodeID u, const BlockID u_block) {
    const auto [target_block, relative_gain] = compute_best_target_block(u, u_block);
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

    const auto [target_block, relative_gain] = compute_best_target_block(u, _p_graph->block(u));
    if (target_block == kInvalidBlockID) {
      _priority_queue.remove(u);
      return;
    }

    _priority_queue.change_priority(u, relative_gain);
    _target_blocks[u] = target_block;
  }

  [[nodiscard]] std::pair<BlockID, RelativeGain>
  compute_best_target_block(const NodeID u, const BlockID u_block) const {
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
      return {kInvalidNodeID, 0};
    }

    const EdgeWeight from_connection = _gain_cache.connection(u, u_block);
    const EdgeWeight absolute_gain = target_block_connection - from_connection;

    const RelativeGain relative_gain = compute_relative_gain(absolute_gain, u_weight);
    return {target_block, relative_gain};
  }

private:
  PartitionedGraph *_p_graph;
  const Graph *_graph;
  std::span<const BlockWeight> _max_block_weights;

  StaticArray<bool> _is_overloaded;
  BlockID _num_overloaded_blocks;

  EdgeWeight _gain;
  GainCache _gain_cache;

  PriorityQueue _priority_queue;
  StaticArray<BlockID> _target_blocks;
};

class SequentialGreedyBalancer : public Refiner {
  template <typename Graph>
  using SequentialGreedyBalancerImpl = SequentialGreedyBalancerImpl<PartitionedGraph, Graph>;

  using SequentialGreedyBalancerCSRImpl = SequentialGreedyBalancerImpl<CSRGraph>;
  using SequentialGreedyBalancerCompressedImpl = SequentialGreedyBalancerImpl<CompressedGraph>;

public:
  SequentialGreedyBalancer(const Context &ctx);
  ~SequentialGreedyBalancer() override;

  SequentialGreedyBalancer &operator=(const SequentialGreedyBalancer &) = delete;
  SequentialGreedyBalancer(const SequentialGreedyBalancer &) = delete;

  SequentialGreedyBalancer &operator=(SequentialGreedyBalancer &&) = default;
  SequentialGreedyBalancer(SequentialGreedyBalancer &&) noexcept = default;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::unique_ptr<SequentialGreedyBalancerCSRImpl> _csr_impl;
  std::unique_ptr<SequentialGreedyBalancerCompressedImpl> _compressed_impl;
};

} // namespace kaminpar::shm
