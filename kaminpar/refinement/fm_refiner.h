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
      _node_pq.emplace_back(_p_ctx.n, _p_ctx.n, _fm._shared_pq_handles.data());
    }
  }

  EdgeWeight run() {
    // Keep track of nodes that we don't want to unlock afterwards
    std::vector<NodeID> committed_moves;

    // Poll seed nodes from the border node arrays
    _fm.poll_border_nodes(
        _fm_ctx.num_seed_nodes,
        _id,
        [&](const NodeID seed_node) {
          insert_into_node_pq(_p_graph, _fm._gain_cache, seed_node);

          // Never unlock seed nodes, even if no move gets committed
          committed_moves.push_back(seed_node);
        }
    );

    // Keep track of all nodes that we touched, so that we can unlock those that
    // have not been moved afterwards
    std::vector<NodeID> touched_nodes;

    // Keep track of the current (expected) gain to decide when to accept a
    // delta partition
    EdgeWeight current_total_gain = 0;
    EdgeWeight best_total_gain = 0;

    while (update_block_pq() && !_stopping_policy.should_stop()) {
      const BlockID block_from = _block_pq.peek_id();
      KASSERT(block_from < _p_graph.k());

      const NodeID node = _node_pq[block_from].peek_id();
      KASSERT(node < _p_graph.n());

      const EdgeWeight expected_gain = _node_pq[block_from].peek_key();
      const auto [block_to, actual_gain] =
          best_gain(_d_graph, _d_gain_cache, node);

      // If the gain got worse, reject the move and try again
      if (actual_gain < expected_gain) {
        _node_pq[block_from].change_priority(node, actual_gain);
        _fm._target_blocks[node] = block_to;
        if (_node_pq[block_from].peek_key() != _block_pq.key(block_from)) {
          _block_pq.change_priority(
              block_from, _node_pq[block_from].peek_key()
          );
        }

        continue;
      }

      // Otherwise, we can remove the node from the PQ
      _node_pq[block_from].pop();
      _fm._locked[node] = -1; // Mark as moved, won't update PQs for this node

      // Skip the move if there is no viable target block
      if (block_to == block_from) {
        continue;
      }

      // Accept the move if the target block does not get overloaded
      const NodeWeight node_weight = _p_graph.node_weight(node);
      if (_d_graph.block_weight(block_to) + node_weight <=
          _p_ctx.block_weights.max(block_to)) {

        // Perform local move
        _d_graph.set_block(node, block_to);
        _d_gain_cache.move(_d_graph, node, block_from, block_to);
        _stopping_policy.update(actual_gain);
        current_total_gain += actual_gain;

        // If we found a new local minimum, apply the moves to the global
        // partition
        if (current_total_gain > best_total_gain) {
          DBG << "Worker " << _id << " committed local improvement with gain "
              << current_total_gain;

          // Update global graph and global gain cache
          for (const auto &[moved_node, moved_to] : _d_graph.delta()) {
            _fm._gain_cache.move(
                _p_graph, moved_node, _p_graph.block(moved_node), moved_to
            );
            _p_graph.set_block(moved_node, moved_to);
            committed_moves.push_back(moved_node);
          }

          // Flush local delta
          _d_graph.clear();
          _d_gain_cache.clear();
          _stopping_policy.reset();

          best_total_gain = current_total_gain;
        }

        for (const auto &[e, v] : _p_graph.neighbors(node)) {
          if (_fm.owner(v) == _id) {
            KASSERT(_node_pq[_p_graph.block(v)].contains(v), "node not in PQ");
            update_after_move(v, node, block_from, block_to);
          } else if (_fm.owner(v) == 0 && _fm.lock_node(v, _id)) {
            insert_into_node_pq(_d_graph, _d_gain_cache, v);
            touched_nodes.push_back(v);
          }
        }
      }
    }

    // Flush local state for the nex tround
    for (auto &node_pq : _node_pq) {
      node_pq.clear();
    }

    _block_pq.clear();
    _d_graph.clear();
    _d_gain_cache.clear();
    _stopping_policy.reset();

    // @todo should be optimized with timestamping

    // Unlock all nodes that were touched, lock the moved ones for good
    // afterwards
    for (const NodeID touched_node : touched_nodes) {
      _fm._locked[touched_node] = 0;
    }

    // ... but keep nodes that we actually moved locked
    for (const NodeID moved_node : committed_moves) {
      _fm._locked[moved_node] = -1;
    }

    return best_total_gain;
  }

private:
  template <typename PartitionedGraphVariant, typename GainCache>
  void insert_into_node_pq(
      const PartitionedGraphVariant &p_graph,
      const GainCache &gain_cache,
      const NodeID u
  ) {
    const BlockID block_u = p_graph.block(u);
    const auto [block_to, gain] = best_gain(p_graph, gain_cache, u);
    KASSERT(!_node_pq[block_u].contains(u), "node already contained in PQ");
    _fm._target_blocks[u] = block_to;
    _node_pq[block_u].push(u, gain);
  }

  void update_after_move(
      const NodeID node,
      const NodeID moved_node,
      const BlockID moved_from,
      const BlockID moved_to
  ) {
    //KASSERT(_d_graph.block(node) == _p_graph.block(node));
    const BlockID old_block = _p_graph.block(node);
    const BlockID old_target_block = _fm._target_blocks[node];

    if (moved_to == old_target_block) {
      // In this case, old_target_block got even better
      // We only need to consider other blocks if old_target_block is full now
      if (_d_graph.block_weight(old_target_block) +
              _d_graph.node_weight(node) <=
          _p_ctx.block_weights.max(old_target_block)) {
        _node_pq[old_block].change_priority(
            node, _d_gain_cache.gain(node, old_block, old_target_block)
        );
      } else {
        const auto [new_target_block, new_gain] =
            best_gain(_d_graph, _d_gain_cache, node);
        _fm._target_blocks[node] = new_target_block;
        _node_pq[old_block].change_priority(node, new_gain);
      }
    } else if (moved_from == old_target_block) {
      // old_target_block go worse, thus have to re-consider all other blocks
      const auto [new_target_block, new_gain] =
          best_gain(_d_graph, _d_gain_cache, node);
      _fm._target_blocks[node] = new_target_block;
      _node_pq[old_block].change_priority(node, new_gain);
    } else if (moved_to == old_block) {
      KASSERT(moved_from != old_target_block);
      // Since we did not move from old_target_block, this block is still the
      // best and we can still move to that block
      _node_pq[old_block].change_priority(
          node, _d_gain_cache.gain(node, old_block, old_target_block)
      );
    } else {
      // old_target_block OR moved_to is best
      const EdgeWeight gain_old_target_block =
          _d_gain_cache.gain(node, old_block, old_target_block);
      const EdgeWeight gain_moved_to =
          _d_gain_cache.gain(node, old_block, moved_to);

      if (gain_moved_to > gain_old_target_block &&
          _d_graph.block_weight(moved_to) + _d_graph.node_weight(node) <=
              _p_ctx.block_weights.max(moved_to)) {
        _fm._target_blocks[node] = moved_to;
        _node_pq[old_block].change_priority(node, gain_moved_to);
      } else {
        _node_pq[old_block].change_priority(node, gain_old_target_block);
      }
    }

    // Check that PQ state is as if we had reconsidered the gains to all blocks
    // This check only works with one thread
    /*
    KASSERT(
        [&] {
          const auto actual = best_gain(_d_graph, _d_gain_cache, node);
          if (_node_pq[old_block].key(node) != actual.second) {
            LOG_WARNING << "node " << node << " has incorrect gain: expected "
                        << actual.second << ", but got "
                        << _node_pq[old_block].key(node);
            return false;
          }
          if (actual.second !=
              _d_gain_cache.gain(node, old_block, _fm._target_blocks[node])) {
            LOG_WARNING << "node " << node << " has incorrect target block";
            return false;
          }
          return true;
        }(),
        "inconsistent PQ state after node move",
        assert::heavy
    );
    */
  }

  template <typename PartitionedGraphVariant, typename GainCache>
  std::pair<BlockID, EdgeWeight> best_gain(
      const PartitionedGraphVariant &p_graph,
      const GainCache &gain_cache,
      const NodeID u
  ) {
    const BlockID block_u = p_graph.block(u);
    const NodeWeight weight_u = p_graph.node_weight(u);

    // Since we use max heaps, we can insert this value into the PQ
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
      if (!_node_pq[block].empty()) {
        const EdgeWeight gain = _node_pq[block].peek_key();
        _block_pq.push_or_change_priority(block, gain);
        have_more_nodes = true;
      } else if (_block_pq.contains(block)) {
        _block_pq.remove(block);
      }
    }
    return have_more_nodes;
  }

  int _id;

  /* Shared data structures */

  FMRefiner &_fm;
  const PartitionContext &_p_ctx;
  const KwayFMRefinementContext &_fm_ctx;
  PartitionedGraph &_p_graph;

  /* Thread-local data structures */

  DeltaPartitionedGraph<false> _d_graph;
  DeltaGainCache<DenseGainCache> _d_gain_cache;
  BinaryMaxHeap<EdgeWeight> _block_pq;
  std::vector<SharedBinaryMaxHeap<EdgeWeight>> _node_pq;

  AdaptiveStoppingPolicy _stopping_policy;
};
} // namespace kaminpar::shm
