/*******************************************************************************
 * @file:   bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Interface for initial partitioning algorithms.
 ******************************************************************************/
#pragma once

#include <array>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::ip {
class Bipartitioner {
public:
  Bipartitioner(const Bipartitioner &) = delete;
  Bipartitioner &operator=(Bipartitioner &&) = delete;

  Bipartitioner(Bipartitioner &&) noexcept = default;
  Bipartitioner &operator=(const Bipartitioner &) = delete;

  virtual ~Bipartitioner() = default;

  virtual void init(const CSRGraph &graph, const PartitionContext &p_ctx) {
    _graph = &graph;
    _p_ctx = &p_ctx;

    KASSERT(_p_ctx->k == 2u, "not a bipartition context", assert::light);
  }

  //! Compute bipartition and return as partitioned graph.
  PartitionedCSRGraph
  bipartition(StaticArray<BlockID> partition, StaticArray<BlockWeight> block_weights) {
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

    return {*_graph, 2, std::move(_partition), std::move(_final_block_weights)};
  }

protected:
  static constexpr BlockID V1 = 0;
  static constexpr BlockID V2 = 1;

  Bipartitioner(const InitialPoolPartitionerContext &pool_ctx) : _pool_ctx(pool_ctx) {}

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
    KASSERT(_partition[u] != kInvalidBlockID, "only use set_block() instead");

    _partition[u] = b;

    const NodeWeight u_weight = _graph->node_weight(u);
    _block_weights[b] += u_weight;
    _block_weights[other_block(b)] -= u_weight;
  }

  inline BlockID other_block(const BlockID b) {
    return 1 - b;
  }

  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;
  const InitialPoolPartitionerContext &_pool_ctx;

  std::array<BlockWeight, 2> _block_weights;

  StaticArray<BlockID> _partition;
  StaticArray<BlockWeight> _final_block_weights;
};
} // namespace kaminpar::shm::ip
