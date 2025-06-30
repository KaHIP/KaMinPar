/**
 * @file partitioned_graph.cc
 * @brief A dynamic graph partition on top of a static graph.
 */
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include <type_traits>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename Graph>
void GenericPartitionedGraph<Graph>::reinit_dense_block_weights(const bool sequentially) {
  if (sequentially) {
    for (const BlockID b : blocks()) {
      _dense_block_weights[b] = 0;
    }

    for (NodeID u = 0, n = _graph->n(); u < n; ++u) {
      const BlockID b = block(u);
      KASSERT(b < _k);

      _dense_block_weights[b] += node_weight(u);
    }
  } else {
    static constexpr BlockID kLowContentionThreshold = 65536;

    tbb::parallel_for<BlockID>(0, k(), [&](const BlockID b) { _dense_block_weights[b] = 0; });

    // If there are lots of block, assume we can sum block weights directly without causing too
    // much contention
    if (k() >= kLowContentionThreshold) {
      tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
        const BlockID b = block(u);
        KASSERT(b != kInvalidBlockID);

        __atomic_fetch_add(&_dense_block_weights[b], node_weight(u), __ATOMIC_RELAXED);
      });

      return;
    }

    // Otherwise, aggregate in thread-local buffers and accumulate afterwards to avoid contention
    tbb::enumerable_thread_specific<StaticArray<BlockWeight>> block_weights_ets([&] {
      return StaticArray<BlockWeight>(k());
    });

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, _graph->n()), [&](const auto &r) {
      StaticArray<BlockWeight> &block_weights = block_weights_ets.local();

      for (NodeID u = r.begin(); u != r.end(); ++u) {
        const BlockID b = block(u);
        KASSERT(b != kInvalidBlockID);

        block_weights[b] += node_weight(u);
      }
    });

    tbb::parallel_for<BlockID>(0, k(), [&](const BlockID b) {
      BlockWeight sum = 0;

      for (const StaticArray<BlockWeight> &block_weights : block_weights_ets) {
        sum += block_weights[b];
      }

      _dense_block_weights[b] = sum;
    });
  }
}

template <typename Graph>
void GenericPartitionedGraph<Graph>::reinit_aligned_block_weights(const bool sequentially) {
  if (!use_aligned_block_weights()) {
    return;
  }

  if (sequentially) {
    for (const BlockID b : blocks()) {
      _aligned_block_weights[b].value = _dense_block_weights[b];
    }
  } else {
    pfor_blocks([&](const BlockID b) {
      _aligned_block_weights[b].value = _dense_block_weights[b];
    });
  }
}

template <typename Graph>
void GenericPartitionedGraph<Graph>::sync_dense_and_aligned_block_weights() const {
  if (!_diverged_block_weights) {
    return;
  }

  // Avoid parallelism in the bipartite case (often used by sequential initial bipartitioning)
  if (_k == 2) {
    _dense_block_weights[0] = _aligned_block_weights[0].value;
    _dense_block_weights[1] = _aligned_block_weights[1].value;
  } else {
    tbb::parallel_for<BlockID>(0, _k, [&](const BlockID b) {
      _dense_block_weights[b] = _aligned_block_weights[b].value;
    });
  }

  _diverged_block_weights = false;
}

template <typename Graph> void GenericPartitionedGraph<Graph>::init_node_weights() {
  if constexpr (std::is_same_v<Graph, CSRGraph>) {
    _node_weights = _graph->raw_node_weights().view();
  } else {
    _node_weights =
        reified(*_graph, [&](const auto &graph) { return graph.raw_node_weights().view(); });
  }
}

template <typename Graph>
void GenericPartitionedGraph<Graph>::alloc_block_weights(const bool sequentially) {
  if (sequentially) {
    if (_dense_block_weights.empty()) {
      _dense_block_weights.resize(_k, static_array::seq);
    }
    if (use_aligned_block_weights()) {
      _aligned_block_weights.resize(_k, static_array::small, static_array::seq);
    }
  } else {
    if (_dense_block_weights.empty()) {
      _dense_block_weights.resize(_k);
    }
    if (use_aligned_block_weights()) {
      _aligned_block_weights.resize(_k);
    }
  }
}

template class GenericPartitionedGraph<Graph>;

template class GenericPartitionedGraph<CSRGraph>;

} // namespace kaminpar::shm
