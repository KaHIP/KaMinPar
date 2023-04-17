/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#include "kaminpar/refinement/fm_refiner.h"

#include <cmath>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/marker.h"
#include "common/parallel/atomic.h"
#include "common/timer.h"

namespace kaminpar::shm {
namespace {
static constexpr std::size_t kNumTouchedNodes = 0;
static constexpr std::size_t kNumCommittedMoves = 1;
static constexpr std::size_t kNumDiscardedMoves = 2;
static constexpr std::size_t kNumRecomputedGains = 3;
static constexpr std::size_t kNumBatches = 4;
static constexpr std::size_t kNumPQInserts = 5;
static constexpr std::size_t kNumPQPops = 7;
static constexpr std::size_t kNumPQUpdates = 6;

using FMStats =
    std::tuple<NodeID, NodeID, NodeID, NodeID, NodeID, NodeID, NodeID>;

struct GlobalFMStats {
  std::vector<FMStats> iteration_stats;

  GlobalFMStats() {
    next_iteration();
  }

  void next_iteration() {
    iteration_stats.emplace_back();
  }

  void reset() {
    iteration_stats.clear();
    next_iteration();
  }
};
} // namespace

FMRefiner::FMRefiner(const Context &ctx)
    : _fm_ctx(ctx.refinement.kway_fm),
      _locked(ctx.partition.n),
      _gain_cache(ctx.partition.k, ctx.partition.n),
      _shared_pq_handles(ctx.partition.n),
      _target_blocks(ctx.partition.n) {
  // Initialize shared PQ handles -- @todo clean this up
  tbb::parallel_for<std::size_t>(
      0,
      _shared_pq_handles.size(),
      [&](std::size_t i) {
        _shared_pq_handles[i] = SharedBinaryMaxHeap<EdgeWeight>::kInvalidID;
      }
  );
}

bool FMRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  SCOPED_TIMER("FM");

  START_TIMER("Initialize gain cache");
  _gain_cache.initialize(*_p_graph);
  STOP_TIMER();

  const EdgeWeight initial_cut = metrics::edge_cut(*_p_graph);

  // Total gain across all iterations
  // This value is not accurate, but the sum of all gains achieved on local
  // delta graphs
  EdgeWeight total_expected_gain = 0;

  for (int iteration = 0; iteration < _fm_ctx.num_iterations; ++iteration) {
    SCOPED_TIMER("Iteration", std::to_string(iteration));

    // Gains of the current iterations
    tbb::enumerable_thread_specific<EdgeWeight> expected_gain_ets;

    // Make sure that we work with correct gains
    KASSERT(
        _gain_cache.validate(*_p_graph),
        "gain cache invalid before iteration " << iteration,
        assert::heavy
    );

    // Find current border nodes
    // This also initializes _locked[], or resets it after the first round
    init_border_nodes();

    LOG << "Starting FM iteration " << iteration << " with "
        << _border_nodes.size() << " border nodes and "
        << tbb::this_task_arena::max_concurrency() << " worker threads";

    // Start one worker per thread
    std::atomic<int> next_id = 0;
    tbb::parallel_for<int>(
        0,
        tbb::this_task_arena::max_concurrency(),
        [&](int) {
          EdgeWeight &expected_gain = expected_gain_ets.local();
          LocalizedFMRefiner localized_fm(
              ++next_id, p_ctx, _fm_ctx, p_graph, *this
          );

          // The workers attempt to extract seed nodes from the border nodes
          // that are still available, continuing this process until there are
          // no more border nodes
          while (has_border_nodes()) {
            expected_gain += localized_fm.run();
          }
        }
    );

    // Abort early if the expected cut improvement falls below the abortion
    // threshold
    // @todo is it feasible to work with these "expected" values? Do we need
    // accurate gains?
    const EdgeWeight expected_gain = expected_gain_ets.combine(std::plus{});
    total_expected_gain += expected_gain;
    const EdgeWeight expected_current_cut = initial_cut - total_expected_gain;
    const EdgeWeight abortion_threshold =
        expected_current_cut * _fm_ctx.improvement_abortion_threshold;
    LOG << "Expected total gain after iteration " << iteration << ": "
        << total_expected_gain
        << ", actual gain: " << initial_cut - metrics::edge_cut(*_p_graph);

    if (expected_gain < abortion_threshold) {
      DBG << "Aborting because expected gain is below threshold "
          << abortion_threshold;
      break;
    }
  }

  return total_expected_gain;
}

void FMRefiner::init_border_nodes() {
  _border_nodes.clear();
  _p_graph->pfor_nodes([&](const NodeID u) {
    if (_gain_cache.is_border_node(u, _p_graph->block(u))) {
      _border_nodes.push_back(u);
    }
    _locked[u] = 0;
  });
  _next_border_node = 0;
}

bool FMRefiner::has_border_nodes() const {
  return _next_border_node < _border_nodes.size();
}

bool FMRefiner::lock_node(const NodeID u, const int id) {
  int free = 0;
  return __atomic_compare_exchange_n(
      &_locked[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
  );
}

int FMRefiner::owner(const NodeID u) {
  return _locked[u];
}

void FMRefiner::unlock_node(const NodeID u) {
  __atomic_store_n(&_locked[u], 0, __ATOMIC_RELAXED);
  // if (_gain_cache.is_border_node(u, _p_graph->block(u))) {
  //   _border_nodes.push_back(u);
  // }
}

EdgeWeight LocalizedFMRefiner::run() {
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
        _block_pq.change_priority(block_from, _node_pq[block_from].peek_key());
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

LocalizedFMRefiner::LocalizedFMRefiner(
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

void LocalizedFMRefiner::update_after_move(
    const NodeID node,
    const NodeID moved_node,
    const BlockID moved_from,
    const BlockID moved_to
) {
  // KASSERT(_d_graph.block(node) == _p_graph.block(node));
  const BlockID old_block = _p_graph.block(node);
  const BlockID old_target_block = _fm._target_blocks[node];

  if (moved_to == old_target_block) {
    // In this case, old_target_block got even better
    // We only need to consider other blocks if old_target_block is full now
    if (_d_graph.block_weight(old_target_block) + _d_graph.node_weight(node) <=
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
  // This is only a valid assertion if we only use one thread.
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

bool LocalizedFMRefiner::update_block_pq() {
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
} // namespace kaminpar::shm
