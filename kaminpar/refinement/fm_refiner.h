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

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

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

  bool lock_node(const NodeID u, const int id);
  int owner(const NodeID u);
  void unlock_node(const NodeID u);

  PartitionedGraph *_p_graph;
  const PartitionContext *_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
  NoinitVector<std::uint8_t> _locked;

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
public:
  LocalizedFMRefiner(
      const int id,
      const PartitionContext &p_ctx,
      const KwayFMRefinementContext &fm_ctx,
      PartitionedGraph &p_graph,
      FMRefiner &fm
  )
      : _id(id),
        _fm(fm),
        _p_ctx(p_ctx),
        _fm_ctx(fm_ctx),
        _p_graph(p_graph),
        _d_graph(&_p_graph),
        _d_gain_cache(_fm._gain_cache),
        _block_pq(_p_ctx.k),
        _stopping_policy(_fm_ctx.alpha) {
    _stopping_policy.init(_p_graph.n());
    for (const BlockID b : _p_graph.blocks()) {
      _vertex_pqs.emplace_back(
          _p_ctx.n, _p_ctx.n, _fm._shared_pq_handles.data()
      );
    }
  }

  EdgeWeight run() {
    // Pool seed nodes from the border node arrays
    _fm.poll_border_nodes(_fm_ctx.num_seed_nodes, [&](const NodeID u) {
      insert_into_pq(_p_graph, _fm._gain_cache, u);
    });

    // Keep track of all nodes that we lock, so that we can unlock them
    // afterwards; do not unlock seed nodes
    std::vector<NodeID> touched_nodes;

    // Keep track of the current (expected) gain to decide when to accept a
    // delta partition
    EdgeWeight total_gain = 0;

    while (update_block_pq() && !_stopping_policy.should_stop()) {
      const BlockID block_from = _block_pq.peek_id();
      const NodeID node = _vertex_pqs[block_from].peek_id();
      const EdgeWeight expected_gain = _vertex_pqs[block_from].peek_key();
      const auto [block_to, actual_gain] =
          best_gain(_d_graph, _d_gain_cache, node);

      // If the gain got worse, reject the move and try again
      if (actual_gain < expected_gain) {
        _vertex_pqs[block_from].change_priority(node, actual_gain);
        _fm._target_blocks[node] = block_to;
        if (_vertex_pqs[block_from].peek_key() != _block_pq.key(block_from)) {
          _block_pq.change_priority(
              block_from, _vertex_pqs[block_from].peek_key()
          );
        }

        continue;
      }

      // Otherwise, we can remove the node from the PQ
      _vertex_pqs[block_from].pop();

      // Accept the move if the target block does not get overloaded
      const NodeWeight node_weight = _p_graph.node_weight(node);
      if (_d_graph.block_weight(block_to) + node_weight <=
          _p_ctx.block_weights.max(block_to)) {

        // Perform local move
        _d_graph.set_block(node, block_to);
        _d_gain_cache.move(_d_graph, node, block_from, block_to);
        _stopping_policy.update(actual_gain);
        total_gain += actual_gain;

        // If we found a new local minimum, apply the moves to the global
        // partition
        if (total_gain > 0) {
          for (const auto &[u, b] : _d_graph.delta()) {
            // Update global graph and global gain cache
            _p_graph.set_block(u, b);
            _fm._gain_cache.move(_p_graph, u, block_from, block_to);

            // Flush local delta
            _d_graph.clear();
            _d_gain_cache.clear();
          }
        }

        for (const auto &[e, v] : _p_graph.neighbors(node)) {
          if (_fm.owner(v) == _id) {
            update_after_move(v, node, block_from, block_to);
          } else if (_fm.owner(v) == 0 && _fm.lock_node(v, _id)) {
            insert_into_pq(_d_graph, _d_gain_cache, v);
            touched_nodes.push_back(v);
          }
        }
      }
    }

    // Unlock all nodes that were touched during this search
    // This does not include seed nodes
    for (const NodeID u : touched_nodes) {
      _fm.unlock_node(u);
    }

    // Flush local state for the nex tround
    for (auto &vertex_pq : _vertex_pqs) {
      vertex_pq.clear();
    }

    _block_pq.clear();
    _d_graph.clear();
    _d_gain_cache.clear();
    _stopping_policy.reset();

    return total_gain;
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

  void update_after_move(
      const NodeID update_node,
      const NodeID moved_node,
      const BlockID block_from,
      const BlockID block_to
  ) {
    const BlockID block_update_node = _d_graph.block(update_node);
    const BlockID old_block_to = _fm._target_blocks[update_node];
    const auto [new_block_to, gain] =
        best_gain(_d_graph, _d_gain_cache, update_node);

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

  bool update_block_pq() {
    bool have_more_nodes = false;
    for (const BlockID block : _p_graph.blocks()) {
      if (!_vertex_pqs[block].empty()) {
        const EdgeWeight gain = _vertex_pqs[block].peek_key();
        _block_pq.push_or_change_priority(block, gain);
        have_more_nodes = true;
      } else {
        _block_pq.remove(block);
      }
    }
    return have_more_nodes;
  }

  // Each worker has a unique ID to lock nodes for its current search
  int _id;

  /* Shared data structures */

  FMRefiner &_fm;
  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;
  PartitionedGraph &_p_graph;

  /* Thread-local data structures */

  DeltaPartitionedGraph _d_graph;
  DeltaGainCache<DenseGainCache> _d_gain_cache;
  BinaryMinHeap<EdgeWeight> _block_pq;
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _vertex_pqs;

  AdaptiveStoppingPolicy _stopping_policy;
};

} // namespace kaminpar::shm
