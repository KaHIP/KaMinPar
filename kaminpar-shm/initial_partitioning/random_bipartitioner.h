/*******************************************************************************
 * Random initial bipartitioner that uses actual PRNG.
 *
 * @file:   random_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/bipartitioner.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::ip {
class RandomBipartitioner : public Bipartitioner {
public:
  explicit RandomBipartitioner(const InitialPartitioningContext &i_ctx) : Bipartitioner(i_ctx) {}

protected:
  void bipartition_impl() override {
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

  Random &_rand = Random::instance();
};
} // namespace kaminpar::shm::ip
