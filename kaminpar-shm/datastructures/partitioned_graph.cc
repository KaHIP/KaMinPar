/*******************************************************************************
 * Dynamic partition wrapper for a static graph.
 *
 * @file:   partitioned_graph.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {
PartitionedGraph::PartitionedGraph(
    const Graph &graph, BlockID k, StaticArray<BlockID> partition, std::vector<BlockID> final_k
)
    : GraphDelegate(&graph),
      _k(k),
      _partition(std::move(partition)),
      _block_weights(k),
      _final_k(std::move(final_k)) {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }
  if (_final_k.empty()) {
    _final_k.resize(k, 1);
  }
  KASSERT(_partition.size() == graph.n());
  init_block_weights_par();
}

PartitionedGraph::PartitionedGraph(
    tag::Sequential,
    const Graph &graph,
    BlockID k,
    StaticArray<BlockID> partition,
    std::vector<BlockID> final_k
)
    : GraphDelegate(&graph),
      _k(k),
      _partition(std::move(partition)),
      _block_weights(k),
      _final_k(std::move(final_k)) {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }
  if (_final_k.empty()) {
    _final_k.resize(k, 1);
  }
  KASSERT(_partition.size() == graph.n());
  init_block_weights_seq();
}

PartitionedGraph::PartitionedGraph(
    NoBlockWeights, const Graph &graph, const BlockID k, StaticArray<BlockID> partition
)
    : GraphDelegate(&graph),
      _k(k),
      _partition(std::move(partition)) {
  if (graph.n() > 0 && _partition.empty()) {
    _partition.resize(_graph->n(), kInvalidBlockID);
  }
  if (_final_k.empty()) {
    _final_k.resize(k, 1);
  }
}

void PartitionedGraph::change_k(const BlockID new_k) {
  _block_weights = StaticArray<BlockWeight>(new_k);
  _final_k.resize(new_k);
  _k = new_k;
}

void PartitionedGraph::reinit_block_weights() {
  pfor_blocks([&](const BlockID b) { _block_weights[b] = 0; });
  init_block_weights_par();
}

void PartitionedGraph::init_block_weights_par() {
  tbb::enumerable_thread_specific<StaticArray<BlockWeight>> block_weights_ets([&] {
    return StaticArray<BlockWeight>(k());
  });

  tbb::parallel_for(tbb::blocked_range(static_cast<NodeID>(0), n()), [&](auto &r) {
    auto &block_weights = block_weights_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      if (const BlockID b = block(u); b != kInvalidBlockID) {
        block_weights[b] += node_weight(u);
      }
    }
  });

  tbb::parallel_for(static_cast<BlockID>(0), k(), [&](const BlockID b) {
    BlockWeight sum = 0;
    for (auto &block_weights : block_weights_ets) {
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
