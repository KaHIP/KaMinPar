/*******************************************************************************
 * Dynamic partition wrapper for a static graph.
 *
 * @file:   partitioned_graph.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {
PartitionedGraph::PartitionedGraph(const Graph &graph, BlockID k, StaticArray<BlockID> partition)
    : GraphDelegate(&graph),
      _k(k),
      _partition(std::move(partition)),
      _block_weights(k) {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }

  KASSERT(_partition.size() == graph.n());

  init_block_weights_par();
}

PartitionedGraph::PartitionedGraph(
    seq, const Graph &graph, BlockID k, StaticArray<BlockID> partition
)
    : GraphDelegate(&graph),
      _k(k),
      _partition(std::move(partition)),
      _block_weights(k) {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }

  KASSERT(_partition.size() == graph.n());

  init_block_weights_seq();
}

void PartitionedGraph::init_block_weights_par() {
  tbb::enumerable_thread_specific<StaticArray<BlockWeight>> block_weights_ets([&] {
    return StaticArray<BlockWeight>(k());
  });

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const tbb::blocked_range<NodeID> &r) {
    auto &block_weights = block_weights_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      if (const BlockID b = block(u); b != kInvalidBlockID) {
        block_weights[b] += node_weight(u);
      }
    }
  });

  tbb::parallel_for<BlockID>(0, k(), [&](const BlockID b) {
    BlockWeight sum = 0;
    for (const StaticArray<BlockWeight> &block_weights : block_weights_ets) {
      sum += block_weights[b];
    }
    _block_weights[b] = sum;
  });
}

void PartitionedGraph::init_block_weights_seq() {
  for (const NodeID u : nodes()) {
    if (const BlockID b = block(u); b != kInvalidBlockID) {
      _block_weights[b] += node_weight(u);
    }
  }
}
} // namespace kaminpar::shm
