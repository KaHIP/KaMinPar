/*******************************************************************************
 * @file:   greedy_graph_growing_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Initial partitioner based on greedy graph growing.
 ******************************************************************************/
#pragma once

#include <kassert/kassert.hpp>

#include "kaminpar-shm/initial_partitioning/bipartitioner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::ip {
struct GreedyGraphGrowingBipartitionerMemoryContext {
  BinaryMinHeap<EdgeWeight> queue{0};
  Marker<> marker{0};

  std::size_t memory_in_kb() const {
    return queue.memory_in_kb() + marker.memory_in_kb();
  }
};

template <typename Graph> class GreedyGraphGrowingBipartitioner : public Bipartitioner<Graph> {
  using MemoryContext = GreedyGraphGrowingBipartitionerMemoryContext;
  using Base = Bipartitioner<Graph>;
  using Base::_block_weights;
  using Base::_graph;
  using Base::_p_ctx;
  using Base::_partition;
  using Base::change_block;
  using Base::V1;
  using Base::V2;

public:
  GreedyGraphGrowingBipartitioner(
      const Graph &graph,
      const PartitionContext &p_ctx,
      const InitialPartitioningContext &i_ctx,
      MemoryContext &m_ctx
  )
      : Bipartitioner<Graph>(graph, p_ctx, i_ctx),
        _queue{m_ctx.queue},
        _marker{m_ctx.marker} {
    if (_queue.capacity() < _graph.n()) {
      _queue.resize(_graph.n());
    }
    if (_marker.size() < _graph.n()) {
      _marker.resize(_graph.n());
    }
  }

protected:
  void bipartition_impl() override {
    KASSERT(_graph.n() > 0u);

    std::fill(_partition.begin(), _partition.end(), V1);
    _block_weights[V1] = _graph.total_node_weight();

    Random &rand = Random::instance();

    do {
      // find random unmarked node -- if too many attempts fail, take the first
      // unmarked node in sequence
      NodeID start_node = 0;
      std::size_t counter = 0;
      do {
        start_node = rand.random_index(0, _graph.n());
        counter++;
      } while (_marker.get(start_node) && counter < 5);
      if (_marker.get(start_node)) {
        start_node = _marker.first_unmarked_element();
      }

      _queue.push(start_node, compute_negative_gain(start_node));
      _marker.set<true>(start_node);

      while (!_queue.empty()) {
        // add max gain node to V2
        const NodeID u = _queue.peek_id();
        KASSERT(_queue.peek_key() == compute_negative_gain(u));
        _queue.pop();
        change_block(u, V2);
        if (_block_weights[V2] >= _p_ctx.block_weights.perfectly_balanced(V2)) {
          break;
        }

        // queue unmarked neighbors / update gains
        _graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
          if (_partition[u] == V2) {
            return; // v already in V2: won't touch this node anymore
          }
          KASSERT(_partition[v] == V1);

          if (_marker.get(v)) {
            KASSERT(_queue.contains(v)); // marked and not in V2: must already be queued
            _queue.decrease_priority_by(v, 2 * _graph.edge_weight(e));
            KASSERT(_queue.key(v) == compute_negative_gain(v));
          } else {
            KASSERT(!_queue.contains(v));
            _queue.push(v, compute_negative_gain(v));
            _marker.set<true>(v);
          }
        });
      }
    } while (_block_weights[V2] < _p_ctx.block_weights.perfectly_balanced(V2));

    _marker.reset();
    _queue.clear();
  }

private:
  [[nodiscard]] EdgeWeight compute_negative_gain(NodeID u) const {
    EdgeWeight gain = 0;
    _graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      gain += (_partition[u] == _partition[v]) ? _graph.edge_weight(e) : -_graph.edge_weight(e);
    });
    return gain;
  }

  BinaryMinHeap<EdgeWeight> &_queue;
  Marker<> &_marker;
};
} // namespace kaminpar::shm::ip
