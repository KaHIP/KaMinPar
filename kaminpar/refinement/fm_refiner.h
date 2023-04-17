/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#pragma once

#include <cmath>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/refiner.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/noinit_vector.h"
#include "common/parallel/atomic.h"

namespace kaminpar::shm {
class FMRefiner : public Refiner {
  SET_DEBUG(false);

  friend class LocalizedFMRefiner;

public:
  FMRefiner(const Context &ctx);

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  bool run_localized_refinement();

  void init_border_nodes();

  template <typename Lambda>
  NodeID poll_border_nodes(const NodeID count, int id, Lambda &&lambda) {
    NodeID polled = 0;

    while (polled < count && _next_border_node < _border_nodes.size()) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to =
          std::min<NodeID>(from + remaining, _border_nodes.size());

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = _border_nodes[current];
        if (lock_node(node, id)) {
          lambda(node);
          ++polled;
        }
      }
    }

    return polled;
  }

  bool has_border_nodes() const;

  bool lock_node(const NodeID u, const int id);
  int owner(const NodeID u);
  void unlock_node(const NodeID u);

  PartitionedGraph *_p_graph;
  const PartitionContext *_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
  NoinitVector<int> _locked;

  DenseGainCache _gain_cache;

  NoinitVector<std::size_t> _shared_pq_handles;
  NoinitVector<BlockID> _target_blocks;
};

struct Move {
  NodeID node;
  BlockID from;
  BlockID to;
};

class LocalizedFMRefiner {
  SET_DEBUG(FMRefiner::kDebug);

public:
  LocalizedFMRefiner(
      int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      PartitionedGraph &p_graph,
      FMRefiner &fm
  );

  EdgeWeight run();

private:
  template <typename PartitionedGraphType, typename GainCacheType>
  void insert_into_node_pq(
      const PartitionedGraphType &p_graph,
      const GainCacheType &gain_cache,
      const NodeID u
  ) {
    const BlockID block_u = p_graph.block(u);
    const auto [block_to, gain] = best_gain(p_graph, gain_cache, u);
    KASSERT(!_node_pq[block_u].contains(u), "node already contained in PQ");
    _fm._target_blocks[u] = block_to;
    _node_pq[block_u].push(u, gain);
  }

  void update_after_move(
      NodeID node, NodeID moved_node, BlockID moved_from, BlockID moved_to
  );

  template <typename PartitionedGraphType, typename GainCacheType>
  std::pair<BlockID, EdgeWeight> best_gain(
      const PartitionedGraphType &p_graph,
      const GainCacheType &gain_cache,
      const NodeID u
  ) {
    const BlockID block_u = p_graph.block(u);
    const NodeWeight weight_u = p_graph.node_weight(u);

    // Since we use max heaps, it is OK to insert this value into the PQ
    EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
    BlockID best_target_block = block_u;
    NodeWeight best_target_block_weight_gap =
        _p_ctx.block_weights.max(block_u) - p_graph.block_weight(block_u);

    for (const BlockID block : p_graph.blocks()) {
      if (block == block_u) {
        continue;
      }

      const NodeWeight target_block_weight =
          p_graph.block_weight(block) + weight_u;
      const NodeWeight max_block_weight = _p_ctx.block_weights.max(block);
      const NodeWeight block_weight_gap =
          max_block_weight - target_block_weight;

      if (block_weight_gap < best_target_block_weight_gap &&
          block_weight_gap < 0) {
        continue;
      }

      const EdgeWeight gain = gain_cache.gain(u, block_u, block);
      if (gain > best_gain ||
          (gain == best_gain && block_weight_gap > best_target_block_weight_gap
          )) {
        best_gain = gain;
        best_target_block = block;
        best_target_block_weight_gap = block_weight_gap;
      }
    }

    return {best_target_block, best_gain};
  }

  bool update_block_pq();

  int _id;

  /* Shared data structures */

  FMRefiner &_fm;
  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;
  PartitionedGraph &_p_graph;

  /* Thread-local data structures */

  DeltaPartitionedGraph<true, true> _d_graph;
  DeltaGainCache<DenseGainCache> _d_gain_cache;
  BinaryMaxHeap<EdgeWeight> _block_pq;
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _node_pq;

  AdaptiveStoppingPolicy _stopping_policy;
};
} // namespace kaminpar::shm
