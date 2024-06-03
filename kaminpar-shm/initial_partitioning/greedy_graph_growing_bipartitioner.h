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
  explicit GreedyGraphGrowingBipartitioner(const InitialPoolPartitionerContext &pool_ctx)
      : Bipartitioner(pool_ctx) {}

  void init(const CSRGraph &graph, const PartitionContext &p_ctx) final;

protected:
  void fill_bipartition() final;

private:
  [[nodiscard]] EdgeWeight compute_gain(NodeID u) const;

  BinaryMinHeap<EdgeWeight> _queue{0};
  Marker<> _marker{0};
};
} // namespace kaminpar::shm::ip
