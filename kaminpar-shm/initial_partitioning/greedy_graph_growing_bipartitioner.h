/*******************************************************************************
 * @file:   greedy_graph_growing_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Initial partitioner based on greedy graph growing.
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/bipartitioner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"

namespace kaminpar::shm::ip {
class GreedyGraphGrowingBipartitioner : public Bipartitioner {
public:
  struct MemoryContext {
    BinaryMinHeap<EdgeWeight> queue{0};
    Marker<> marker{0};

    std::size_t memory_in_kb() const {
      return queue.memory_in_kb() + marker.memory_in_kb();
    }
  };

  GreedyGraphGrowingBipartitioner(const InitialPartitioningContext &i_ctx, MemoryContext &m_ctx)
      : Bipartitioner(i_ctx),
        _queue(m_ctx.queue),
        _marker(m_ctx.marker) {}

  void init(const CSRGraph &graph, const PartitionContext &p_ctx) override {
    Bipartitioner::init(graph, p_ctx);

    if (_queue.capacity() < _graph->n()) {
      _queue.resize(_graph->n());
    }
    if (_marker.size() < _graph->n()) {
      _marker.resize(_graph->n());
    }
  }

protected:
  void bipartition_impl() override;

private:
  [[nodiscard]] EdgeWeight compute_negative_gain(NodeID u) const;

  BinaryMinHeap<EdgeWeight> &_queue;
  Marker<> &_marker;
};
} // namespace kaminpar::shm::ip
