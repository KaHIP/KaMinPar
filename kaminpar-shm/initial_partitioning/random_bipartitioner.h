/*******************************************************************************
 * @file:   random_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Initial partitioner that assigns nodes to random blocks.
 ******************************************************************************/
#pragma once

#include <array>

#include "kaminpar-shm/initial_partitioning/bipartitioner.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::ip {
class RandomBipartitioner : public Bipartitioner {
public:
  struct MemoryContext {
    std::size_t memory_in_kb() const {
      return 0;
    }
  };

  RandomBipartitioner(const Graph &graph, const PartitionContext &p_ctx, const InitialPartitioningContext &i_ctx, MemoryContext &)
      : Bipartitioner(graph, p_ctx, i_ctx) {}

protected:
  void bipartition_impl() override {
    for (const NodeID u : _graph.nodes()) {
      const auto block = _rand.random_index(0, 2);
      if (_block_weights[block] + _graph.node_weight(u) <
          _p_ctx.block_weights.perfectly_balanced(block)) {
        set_block(u, block);
      } else {
        add_to_smaller_block(u);
      }
    }
  }

  Random &_rand = Random::instance();
};
} // namespace kaminpar::shm::ip
