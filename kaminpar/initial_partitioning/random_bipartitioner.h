/*******************************************************************************
 * @file:   random_bipartitioner.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Initial partitioner that assigns nodes to random blocks.
 ******************************************************************************/
#pragma once

#include "kaminpar/initial_partitioning/i_bipartitioner.h"
#include "kaminpar/utility/random.h"

#include <array>

namespace kaminpar {
class RandomBipartitioner : public Bipartitioner {
public:
  struct MemoryContext {
    std::size_t memory_in_kb() const { return 0; }
  };

  RandomBipartitioner(const Graph &graph, const PartitionContext &p_ctx, const InitialPartitioningContext &i_ctx,
                      MemoryContext &)
      : Bipartitioner(graph, p_ctx, i_ctx) {}

protected:
  void bipartition_impl() override {
    for (const NodeID u : _graph.nodes()) {
      const auto block = _rand.random_index(0, 2);
      if (_block_weights[block] + _graph.node_weight(u) < _p_ctx.perfectly_balanced_block_weight(block)) {
        set_block(u, block);
      } else {
        add_to_smaller_block(u);
      }
    }
  }

  Randomize &_rand{Randomize::instance()};
};
} // namespace kaminpar