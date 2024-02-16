/*******************************************************************************
 * @file:   greedy_graph_growing_bipartitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Initial partitioner using greedy graph growing.
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/greedy_graph_growing_bipartitioner.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm::ip {
void GreedyGraphGrowingBipartitioner::bipartition_impl() {
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
      for (const auto [e, v] : _graph.neighbors(u)) {
        if (_partition[u] == V2)
          continue; // v already in V2: won't touch this node anymore
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
      }
    }
  } while (_block_weights[V2] < _p_ctx.block_weights.perfectly_balanced(V2));

  _marker.reset();
  _queue.clear();
}

[[nodiscard]] EdgeWeight GreedyGraphGrowingBipartitioner::compute_negative_gain(const NodeID u
) const {
  EdgeWeight gain = 0;
  for (const auto [e, v] : _graph.neighbors(u)) {
    gain += (_partition[u] == _partition[v]) ? _graph.edge_weight(e) : -_graph.edge_weight(e);
  }
  return gain;
}
} // namespace kaminpar::shm::ip
