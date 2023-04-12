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
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/refiner.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/noinit_vector.h"
#include "common/parallel/atomic.h"

namespace kaminpar::shm {
class FMRefiner : public Refiner {
  friend class LocalizedFMRefiner;

public:
  FMRefiner(const Context &ctx);

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  void initialize(const Graph &graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  [[nodiscard]] EdgeWeight expected_total_gain() const final {
    return 0;
  }

private:
  bool run_localized_refinement();

  void init_border_nodes();

  template <typename Lambda>
  NodeID poll_border_nodes(const NodeID count, Lambda &&lambda) {
    NodeID polled = 0;
    while (polled < count && _next_border_node < _border_nodes.size()) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to =
          std::min<NodeID>(from + remaining, _border_nodes.size());

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = _border_nodes[from];
        std::uint8_t free = 0;
        if (__atomic_compare_exchange_n(
                &_locked[node],
                &free,
                1,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST
            )) {
          lambda(node);
          ++polled;
        }
      }
    }

    return polled;
  }

  bool has_border_nodes() const;

  bool lock_node(const NodeID u);
  void unlock_node(const NodeID u);

  PartitionedGraph *_p_graph;
  const PartitionContext *_p_ctx;
  const KwayFMRefinementContext *_fm_ctx;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
  NoinitVector<std::uint8_t> _locked;

  DenseGainCache _gain_cache;

  NoinitVector<std::size_t> _shared_pq_handles;
  tbb::enumerable_thread_specific<BinaryMinHeap<EdgeWeight>> _pq_ets;
  tbb::enumerable_thread_specific<Marker<>> _marker_ets;
  NoinitVector<BlockID> _target_blocks;
};

struct Move {
  NodeID node;
  BlockID from;
  BlockID to;
};

class LocalizedFMRefiner {
public:
  LocalizedFMRefiner(
      const Context &ctx, PartitionedGraph &p_graph, FMRefiner &fm
  )
      : _fm(fm),
        _p_ctx(ctx.partition),
        _fm_ctx(ctx.refinement.kway_fm),
        _p_graph(p_graph),
        _block_pq(ctx.partition.k),
        _vertex_pqs(
            ctx.partition.k, BinaryMaxHeap<EdgeWeight>(_fm._shared_pq_handles)
        ),
        _stopping_policy(_fm_ctx.alpha) {
    _stopping_policy.init(_p_graph.n());
  }

  EdgeWeight run() {
    DeltaPartitionedGraph d_graph(&_p_graph);
    DeltaGainCache d_gain_cache(_fm._gain_cache, d_graph);

    std::vector<NodeID> touched_nodes;

    _fm.poll_border_nodes(_fm_ctx.num_seed_nodes, [&](const NodeID u) {
      insert_into_pq(_p_graph, _fm._gain_cache, u);
      touched_nodes.push_back(u);
    });

    EdgeWeight total_gain = 0;
    std::vector<Move> moves;

    _stopping_policy.reset();

    while (!_block_pq.empty() && !_stopping_policy.should_stop()) {
      const NodeID node = pq.peek_id();
      const EdgeWeight gain = pq.peek_key();
      const BlockID to = _target_blocks[node];
      const NodeWeight node_weight = _p_graph->node_weight(node);
      pq.pop();

      // If the actual gain is worse than the PQ gain, recompute
      if (gain > _gain_cache.gain_to(node, to)) {
        const auto [block, gain] =
            _gain_cache.best_gain(node, _p_ctx->block_weights);
        pq.push(node, gain);
        _target_blocks[node] = block;

        // ... and try again
        continue;
      }

      // Otherwise, try to move the node
      if (d_graph.block_weight(to) + node_weight <=
          _p_ctx->block_weights.max(to)) {
        total_gain += gain;
        moves.push_back({node, d_graph.block(node), to});
        d_graph.set_block(node, to);
        stopping_policy.update(gain);

        if (total_gain > 0) {
          for (const auto &[u, b] : d_graph.delta()) {
            _gain_cache.move_node(u, _p_graph->block(u), b);
            _p_graph->set_block(u, b);
          }
          moves.clear();
        }

        for (const auto &[e, v] : _p_graph->neighbors(node)) {
          if (!lock_node(v)) {
            continue;
          }

          const auto [block, gain] =
              _gain_cache.best_gain(v, _p_ctx->block_weights);
          if (pq.contains(v)) {
            pq.change_priority(v, gain);
          } else {
            touched_nodes.push_back(v);
            pq.push(v, gain);
          }
          _target_blocks[v] = block;
        }
      } else {
      }
    }

    for (const NodeID u : touched_nodes) {
      unlock_node(u);
    }

    return total_gain > 0;
  }

private:
  template <typename PartitionedGraphVariant, typename GainCache>
  void insert_into_pq(
      const PartitionedGraphVariant &p_graph,
      const GainCache &gain_cache,
      const NodeID u
  ) {
    const BlockID block_u = p_graph.block(u);
    const auto [block_to, gain] = best_gain(p_graph, gain_cache, u);
    _fm._target_blocks[u] = block_to;
    _vertex_pqs[block_u].push(u, gain);
  }

  template <typename PartitionedGraphVariant, typename GainCache>
  void update_after_move(
      const PartitionedGraphVariant &p_graph,
      const GainCache &gain_cache,
      const NodeID update_node,
      const NodeID moved_node,
      const BlockID block_from,
      const BlockID block_to
  ) {
    const BlockID block_update_node = p_graph.block(update_node);
    const BlockID old_block_to = _fm._target_blocks[update_node];
    const auto [new_block_to, gain] =
        best_gain(p_graph, gain_cache, update_node);

    _fm._target_blocks[update_node] = new_block_to;
    _vertex_pqs[block_update_node].change_priority(update_node, gain);
  }

  template <typename PartitionedGraphVariant, typename GainCache>
  std::pair<BlockID, EdgeWeight> best_gain(
      const PartitionedGraphVariant &p_graph,
      const GainCache &gain_cache,
      const NodeID u
  ) {
    const BlockID block_u = p_graph.block(u);
    const NodeWeight weight_u = p_graph.node_weight(u);

    EdgeWeight best_gain = 0;
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

  void update_block_pq() {
    for (const BlockID block : _p_graph.blocks()) {
      if (_vertex_pqs[block].empty()) {
        const EdgeWeight gain = _vertex_pqs[block].peek_key();
        _block_pq.push_or_update(block, gain);
      } else {
        _block_pq.remove(block);
      }
    }
  }

  FMRefiner &_fm;
  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;
  PartitionedGraph &_p_graph;

  BinaryMinHeap<EdgeWeight> _block_pq;
  std::vector<BinaryMaxHeap<EdgeWeight>> _vertex_pqs;

  AdaptiveStoppingPolicy _stopping_policy;
};

} // namespace kaminpar::shm
