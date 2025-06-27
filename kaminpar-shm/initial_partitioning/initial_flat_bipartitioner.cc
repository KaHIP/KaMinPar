/*******************************************************************************
 * Interface for initial bipartitioning algorithms.
 *
 * @file:   initial_flat_bipartitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_flat_bipartitioner.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

void InitialFlatBipartitioner::init(const CSRGraph &graph, const PartitionContext &p_ctx) {
  KASSERT(p_ctx.k == 2u, "must be initialized with a 2-way partition context");

  _graph = &graph;
  _p_ctx = &p_ctx;
}

PartitionedCSRGraph InitialFlatBipartitioner::bipartition(
    StaticArray<BlockID> partition, StaticArray<BlockWeight> block_weights
) {
  if (_graph->n() == 0) {
    block_weights[0] = 0;
    block_weights[1] = 0;

    return {*_graph, 2, std::move(partition), std::move(block_weights)};
  }

  _partition = std::move(partition);
  if (_partition.size() < _graph->n()) {
    _partition.resize(_graph->n(), static_array::seq);
  }
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  std::fill(_partition.begin(), _partition.begin() + _graph->n(), kInvalidBlockID);
#endif

  _final_block_weights = std::move(block_weights);
  if (_final_block_weights.size() < 2) {
    _final_block_weights.resize(2, static_array::seq);
  }

  _block_weights[0] = 0;
  _block_weights[1] = 0;

  fill_bipartition();

  _final_block_weights[0] = _block_weights[0];
  _final_block_weights[1] = _block_weights[1];

  return {
      PartitionedCSRGraph::seq{},
      *_graph,
      2,
      std::move(_partition),
      std::move(_final_block_weights),
  };
}

} // namespace kaminpar::shm
