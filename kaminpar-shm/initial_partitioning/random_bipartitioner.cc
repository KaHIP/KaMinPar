/*******************************************************************************
 * Random initial bipartitioner that uses actual PRNG.
 *
 * @file:   random_bipartitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/random_bipartitioner.h"

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::ip {
RandomBipartitioner::RandomBipartitioner(const InitialPoolPartitionerContext &pool_ctx)
    : Bipartitioner(pool_ctx) {}

void RandomBipartitioner::fill_bipartition() {
  for (const NodeID u : _graph->nodes()) {
    const std::size_t block = _rand.random_index(0, 2);

    if (_block_weights[block] + _graph->node_weight(u) <
        _p_ctx->block_weights.perfectly_balanced(block)) {
      set_block(u, block);
    } else {
      add_to_smaller_block(u);
    }
  }
}
} // namespace kaminpar::shm::ip
