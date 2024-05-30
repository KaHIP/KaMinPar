/*******************************************************************************
 * Initial bipartitioner based on greedy graph growing.
 *
 * @file:   greedy_graph_growing_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/bipartitioner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"

namespace kaminpar::shm::ip {
class GreedyGraphGrowingBipartitioner : public Bipartitioner {
public:
  explicit GreedyGraphGrowingBipartitioner(const InitialPartitioningContext &i_ctx)
      : Bipartitioner(i_ctx) {}

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

  BinaryMinHeap<EdgeWeight> _queue{0};
  Marker<> _marker{0};
};
} // namespace kaminpar::shm::ip
