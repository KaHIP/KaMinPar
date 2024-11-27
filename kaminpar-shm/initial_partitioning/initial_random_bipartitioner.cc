/*******************************************************************************
 * Random initial bipartitioner that uses actual PRNG.
 *
 * @file:   initial_random_bipartitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_random_bipartitioner.h"

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

InitialRandomBipartitioner::InitialRandomBipartitioner(const InitialPoolPartitionerContext &pool_ctx
)
    : InitialFlatBipartitioner(pool_ctx) {}

void InitialRandomBipartitioner::fill_bipartition() {
  for (const NodeID u : _graph->nodes()) {
    const std::size_t block = _rand.random_index(0, 2);

    if (_block_weights[block] + _graph->node_weight(u) <
        _p_ctx->perfectly_balanced_block_weight(block)) {
      set_block(u, block);
    } else {
      add_to_smaller_block(u);
    }
  }
}

} // namespace kaminpar::shm
