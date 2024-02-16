/*******************************************************************************
 * @file:   bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Interface for initial partitioning algorithms.
 ******************************************************************************/
#pragma once

#include <array>
#include <tuple>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::ip {
class Bipartitioner {
public:
  using BlockWeights = std::array<BlockWeight, 2>;

  Bipartitioner(const Bipartitioner &) = delete;
  Bipartitioner &operator=(Bipartitioner &&) = delete;
  Bipartitioner(Bipartitioner &&) noexcept = default;
  Bipartitioner &operator=(const Bipartitioner &) = delete;
  virtual ~Bipartitioner() = default;

  //! Compute bipartition and return as partitioned graph.
  virtual PartitionedGraph bipartition(StaticArray<BlockID> &&partition = {}) {
    return PartitionedGraph(
        PartitionedGraph::seq{}, _graph, 2, bipartition_raw(std::move(partition))
    );
  }

  //! Compute bipartition and return as array.
  StaticArray<BlockID> bipartition_raw(StaticArray<BlockID> &&partition = {}) {
    if (_graph.n() == 0) {
      return {};
    }

    _partition = std::move(partition);
    if (_partition.size() < _graph.n()) {
      _partition.resize(_graph.n());
    }
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    std::fill(_partition.begin(), _partition.end(), kInvalidBlockID);
#endif

    _block_weights.fill(0);
    bipartition_impl();

    return std::move(_partition);
  }

protected:
  static constexpr BlockID V1 = 0;
  static constexpr BlockID V2 = 1;

  Bipartitioner(
      const Graph &graph, const PartitionContext &p_ctx, const InitialPartitioningContext &i_ctx
  )
      : _graph{graph},
        _p_ctx{p_ctx},
        _i_ctx{i_ctx} {
    KASSERT(_p_ctx.k == 2u, "not a bipartition context", assert::light);
  }

  // must be implemented by the base class -- compute bipartition
  virtual void bipartition_impl() = 0;

  //
  // Auxiliary functions for bipartitioning
  //

  inline void add_to_smaller_block(const NodeID u) {
    const NodeWeight delta1{_block_weights[0] - _p_ctx.block_weights.perfectly_balanced(0)};
    const NodeWeight delta2{_block_weights[1] - _p_ctx.block_weights.perfectly_balanced(1)};
    const BlockID block{delta1 < delta2 ? V1 : V2};
    set_block(u, block);
  }

  inline void set_block(const NodeID u, const BlockID b) {
    KASSERT(_partition[u] == kInvalidBlockID, "use update_block() instead");
    _partition[u] = b;
    _block_weights[b] += _graph.node_weight(u);
  }

  inline void change_block(const NodeID u, const BlockID b) {
    KASSERT(_partition[u] != kInvalidBlockID, "only use set_block() instead");
    _partition[u] = b;

    const NodeWeight u_weight = _graph.node_weight(u);
    _block_weights[b] += u_weight;
    _block_weights[other_block(b)] -= u_weight;
  }

  inline BlockID other_block(const BlockID b) {
    return 1 - b;
  }

  const Graph &_graph;
  const PartitionContext &_p_ctx;
  const InitialPartitioningContext &_i_ctx;

  StaticArray<BlockID> _partition;
  BlockWeights _block_weights;
};
} // namespace kaminpar::shm::ip
