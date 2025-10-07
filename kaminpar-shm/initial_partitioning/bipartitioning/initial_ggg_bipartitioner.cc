/*******************************************************************************
 * Initial bipartitioner based on greedy graph growing.
 *
 * @file:   initial_ggg_bipartitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/bipartitioning/initial_ggg_bipartitioner.h"

#include <algorithm>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

void InitialGGGBipartitioner::init(const CSRGraph &graph, const PartitionContext &p_ctx) {
  InitialFlatBipartitioner::init(graph, p_ctx);

  if (_queue.capacity() < _graph->n()) {
    _queue.resize(_graph->n());
  }

  if (_marker.size() < _graph->n()) {
    _marker.resize(_graph->n());
  }
}

void InitialGGGBipartitioner::fill_bipartition() {
  KASSERT(_graph->n() > 0u);

  _marker.reset();
  _queue.clear();

  std::fill(_partition.begin(), _partition.begin() + _graph->n(), V1);
  _block_weights[V1] = _graph->total_node_weight();

  Random &rand = Random::instance();

  do {
    // Find random unmarked node -- if too many attempts fail, take the first
    // unmarked node in sequence
    NodeID start_node = 0;
    std::size_t counter = 0;
    do {
      start_node = rand.random_index(0, _graph->n());
      counter++;
    } while (_marker.get(start_node) && counter < 5);
    if (_marker.get(start_node)) {
      start_node = _marker.first_unmarked_element();
    }
    if (start_node >= _graph->n()) {
      break;
    }

    _queue.push(start_node, compute_gain(start_node));
    _marker.set<true>(start_node);

    while (!_queue.empty()) {
      // Add max gain node to V2
      const NodeID u = _queue.peek_id();
      KASSERT(_queue.peek_key() == compute_gain(u), "invalid gain in queue", assert::heavy);
      _queue.pop();
      change_block(u, V2);
      if (_block_weights[V2] >= _p_ctx->perfectly_balanced_block_weight(V2)) {
        break;
      }

      // Queue unmarked neighbors / update gains
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        if (_partition[u] == V2) {
          // v already in V2: won't touch this node anymore
          return;
        }

        KASSERT(_partition[v] == V1);

        if (_marker.get(v)) {
          // Marked and not in V2: must already be queued
          KASSERT(_queue.contains(v));
          _queue.decrease_priority_by(v, 2 * w);
          KASSERT(_queue.key(v) == compute_gain(v), "invalid gain in queue", assert::heavy);
        } else {
          KASSERT(!_queue.contains(v));
          _queue.push(v, compute_gain(v));
          _marker.set<true>(v);
        }
      });
    }
  } while (_block_weights[V2] < _p_ctx->perfectly_balanced_block_weight(V2));
}

[[nodiscard]] EdgeWeight InitialGGGBipartitioner::compute_gain(const NodeID u) const {
  EdgeWeight gain = 0;

  _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
    if (_partition[u] == _partition[v]) {
      gain += w;
    } else {
      gain -= w;
    }
  });

  return gain;
}

} // namespace kaminpar::shm
