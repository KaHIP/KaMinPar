/*******************************************************************************
 * Initial bipartitioner based on greedy graph growing.
 *
 * @file:   initial_ggg_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/bipartitioning/initial_flat_bipartitioner.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/marker.h"

namespace kaminpar::shm {

class InitialGGGBipartitioner : public InitialFlatBipartitioner {
public:
  explicit InitialGGGBipartitioner(const InitialPoolPartitionerContext &pool_ctx)
      : InitialFlatBipartitioner(pool_ctx) {}

  void init(const CSRGraph &graph, const PartitionContext &p_ctx) final;

protected:
  void fill_bipartition() final;

private:
  [[nodiscard]] EdgeWeight compute_gain(NodeID u) const;

  BinaryMinHeap<EdgeWeight> _queue{0};
  Marker<> _marker{0};
};

} // namespace kaminpar::shm
