/*******************************************************************************
 * Interface for initial bipartitioning algorithms.
 *
 * @file:   initial_flat_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <array>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {
class InitialFlatBipartitioner {
public:
  InitialFlatBipartitioner(const InitialFlatBipartitioner &) = delete;
  InitialFlatBipartitioner &operator=(InitialFlatBipartitioner &&) = delete;

  InitialFlatBipartitioner(InitialFlatBipartitioner &&) noexcept = default;
  InitialFlatBipartitioner &operator=(const InitialFlatBipartitioner &) = delete;

  virtual ~InitialFlatBipartitioner() = default;

  virtual void init(const CSRGraph &graph, const PartitionContext &p_ctx);

  PartitionedCSRGraph
  bipartition(StaticArray<BlockID> partition, StaticArray<BlockWeight> block_weights);

protected:
  static constexpr BlockID V1 = 0;
  static constexpr BlockID V2 = 1;

  InitialFlatBipartitioner(const InitialPoolPartitionerContext &pool_ctx) : _pool_ctx(pool_ctx) {}

  virtual void fill_bipartition() = 0;

  //
  // Auxiliary functions for bipartitioning
  //

  inline void add_to_smaller_block(const NodeID u) {
    const NodeWeight delta1 = _block_weights[0] - _p_ctx->block_weights.perfectly_balanced(0);
    const NodeWeight delta2 = _block_weights[1] - _p_ctx->block_weights.perfectly_balanced(1);
    const BlockID block = delta1 < delta2 ? V1 : V2;
    set_block(u, block);
  }

  inline void set_block(const NodeID u, const BlockID b) {
    KASSERT(_partition[u] == kInvalidBlockID, "use update_block() instead");

    _partition[u] = b;
    _block_weights[b] += _graph->node_weight(u);
  }

  inline void change_block(const NodeID u, const BlockID b) {
    KASSERT(_partition[u] != kInvalidBlockID, "use set_block() instead");

    _partition[u] = b;

    const NodeWeight u_weight = _graph->node_weight(u);
    _block_weights[b] += u_weight;
    _block_weights[other_block(b)] -= u_weight;
  }

  [[nodiscard]] inline BlockID other_block(const BlockID b) {
    return 1 - b;
  }

  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;
  const InitialPoolPartitionerContext &_pool_ctx;

  std::array<BlockWeight, 2> _block_weights;

  StaticArray<BlockID> _partition;
  StaticArray<BlockWeight> _final_block_weights;
};
} // namespace kaminpar::shm
