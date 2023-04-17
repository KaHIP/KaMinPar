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
SET_DEBUG(true);
SET_STATISTICS(true);
} // namespace

namespace {
struct Stats {
  parallel::Atomic<NodeID> num_touched_nodes;
  parallel::Atomic<NodeID> num_committed_moves;
  parallel::Atomic<NodeID> num_discarded_moves;
  parallel::Atomic<NodeID> num_recomputed_gains;
  parallel::Atomic<NodeID> num_batches;
  parallel::Atomic<NodeID> num_pq_inserts;
  parallel::Atomic<NodeID> num_pq_updates;
  parallel::Atomic<NodeID> num_pq_pops;

  Stats &operator+=(const Stats &other) {
    num_touched_nodes += other.num_touched_nodes;
    num_committed_moves += other.num_committed_moves;
    num_discarded_moves += other.num_discarded_moves;
    num_recomputed_gains += other.num_recomputed_gains;
    num_batches += other.num_batches;
    num_pq_inserts += other.num_pq_inserts;
    num_pq_updates += other.num_pq_updates;
    num_pq_pops += other.num_pq_pops;
    return *this;
  }
};

struct GlobalStats {
  std::vector<Stats> iteration_stats;

  GlobalStats() {
    next_iteration();
  }

  void add(const Stats &stats) {
    iteration_stats.back() += stats;
  }

  void next_iteration() {
    iteration_stats.emplace_back();
  }

  void reset() {
    iteration_stats.clear();
    next_iteration();
  }

  void summarize() {
    LOG_STATS << "FM Refinement:";
    for (std::size_t i = 0; i < iteration_stats.size(); ++i) {
      const Stats &stats = iteration_stats[i];

      LOG_STATS << "  * Iteration " << (i + 1) << " of "
                << iteration_stats.size() << ":";
      LOG_STATS << "    + Number of batches: " << stats.num_batches;
      LOG_STATS << "    + Number of touched nodes: " << stats.num_touched_nodes
                << " in total, " << stats.num_touched_nodes / stats.num_batches
                << " per batch";
      LOG_STATS << "    + Number of moves: " << stats.num_committed_moves
                << " committed, " << stats.num_discarded_moves
                << " discarded (= "
                << 100.0 * stats.num_discarded_moves /
                       (stats.num_committed_moves + stats.num_discarded_moves)
                << "%)";
      LOG_STATS << "    + Number of recomputed gains: "
                << stats.num_recomputed_gains;
      LOG_STATS << "    + Number of PQ operations: " << stats.num_pq_inserts
                << " inserts, " << stats.num_pq_updates << " updates, "
                << stats.num_pq_pops << " pops";
    }
  }
};
} // namespace

namespace fm {
class NodeTracker {
public:
  NodeTracker(const NodeID max_n) : _state(max_n) {
    tbb::parallel_for<std::size_t>(0, max_n, [&](const std::size_t i) {
      _state[i] = 0;
    });
  }

  bool lock(const NodeID u, const int id) {
    int free = 0;
    return __atomic_compare_exchange_n(
        &_state[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
    );
  }

  int owner(const NodeID u) const {
    return __atomic_load_n(&_state[u], __ATOMIC_RELAXED);
  }

  void unlock(const NodeID u) {
    __atomic_store_n(&_state[u], 0, __ATOMIC_RELAXED);
  }

  // Generic non-atomic setter.
  // @todo Build a better interface once the details are settled.
  void set(const NodeID node, const int value) {
    _state[node] = value;
  }

private:
  NoinitVector<int> _state;
};

class BorderNodes {
public:
  BorderNodes(DenseGainCache &gain_cache, NodeTracker &node_tracker)
      : _gain_cache(gain_cache),
        _node_tracker(node_tracker) {}

  void init(const PartitionedGraph &p_graph) {
    _border_nodes.clear();
    p_graph.pfor_nodes([&](const NodeID u) {
      if (_gain_cache.is_border_node(u, p_graph.block(u))) {
        _border_nodes.push_back(u);
      }
      _node_tracker.set(u, 0);
    });
    _next_border_node = 0;
  }

  template <typename Lambda>
  NodeID poll(const NodeID count, int id, Lambda &&lambda) {
    NodeID polled = 0;

    while (polled < count && _next_border_node < _border_nodes.size()) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to =
          std::min<NodeID>(from + remaining, _border_nodes.size());

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = _border_nodes[current];
        if (_node_tracker.lock(node, id)) {
          lambda(node);
          ++polled;
        }
      }
    }

    return polled;
  }

  [[nodiscard]] bool has_more() const {
    return _next_border_node < _border_nodes.size();
  }

  [[nodiscard]] std::size_t size() const {
    return _border_nodes.size();
  }

private:
  DenseGainCache &_gain_cache;
  NodeTracker &_node_tracker;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
};

struct SharedData {
  SharedData(const NodeID max_n, const BlockID max_k)
      : node_tracker(max_n),
        gain_cache(max_k, max_n),
        border_nodes(gain_cache, node_tracker),
        shared_pq_handles(max_n),
        target_blocks(max_n) {
    tbb::parallel_for<std::size_t>(
        0,
        shared_pq_handles.size(),
        [&](std::size_t i) {
          shared_pq_handles[i] = SharedBinaryMaxHeap<EdgeWeight>::kInvalidID;
        }
    );
  }

  NodeTracker node_tracker;
  DenseGainCache gain_cache;
  BorderNodes border_nodes;
  NoinitVector<std::size_t> shared_pq_handles;
  NoinitVector<BlockID> target_blocks;
  GlobalStats stats;
};
} // namespace fm

FMRefiner::FMRefiner(const Context &input_ctx)
    : _fm_ctx(&input_ctx.refinement.kway_fm),
      _shared(std::make_unique<fm::SharedData>(
          input_ctx.partition.n, input_ctx.partition.k
      )) {}

FMRefiner::~FMRefiner() = default;

bool FMRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("FM");

  START_TIMER("Initialize gain cache");
  _shared->gain_cache.initialize(p_graph);
  STOP_TIMER();

  const EdgeWeight initial_cut = metrics::edge_cut(p_graph);

  // Total gain across all iterations
  // This value is not accurate, but the sum of all gains achieved on local
  // delta graphs
  EdgeWeight total_expected_gain = 0;

  for (int iteration = 0; iteration < _fm_ctx->num_iterations; ++iteration) {
    // Gains of the current iterations
    tbb::enumerable_thread_specific<EdgeWeight> expected_gain_ets;

    // Make sure that we work with correct gains
    KASSERT(
        _shared->gain_cache.validate(p_graph),
        "gain cache invalid before iteration " << iteration,
        assert::heavy
    );

    // Find current border nodes
    _shared->border_nodes.init(p_graph);

    DBG << "Starting FM iteration " << iteration << " with "
        << _shared->border_nodes.size() << " border nodes and "
        << tbb::this_task_arena::max_concurrency() << " worker threads";

    // Start one worker per thread
    std::atomic<int> next_id = 0;
    tbb::parallel_for<int>(
        0,
        tbb::this_task_arena::max_concurrency(),
        [&](int) {
          EdgeWeight &expected_gain = expected_gain_ets.local();
          LocalizedFMRefiner localized_fm(
              ++next_id, p_ctx, *_fm_ctx, p_graph, *_shared
          );

          // The workers attempt to extract seed nodes from the border nodes
          // that are still available, continuing this process until there are
          // no more border nodes
          while (_shared->border_nodes.has_more()) {
            expected_gain += localized_fm.run_batch();
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
        expected_current_cut * _fm_ctx->improvement_abortion_threshold;
    DBG << "Expected total gain after iteration " << iteration << ": "
        << total_expected_gain
        << ", actual gain: " << initial_cut - metrics::edge_cut(p_graph);

    if (expected_gain < abortion_threshold) {
      DBG << "Aborting because expected gain is below threshold "
          << abortion_threshold;
      break;
    }

    IFSTATS(_shared->stats.next_iteration());
  }

  IFSTATS(_shared->stats.summarize());
  IFSTATS(_shared->stats.reset());

  return total_expected_gain;
}

LocalizedFMRefiner::LocalizedFMRefiner(
    const int id,
    const PartitionContext &p_ctx,
    const KwayFMRefinementContext &fm_ctx,
    PartitionedGraph &p_graph,
    fm::SharedData &shared
)
    : _id(id),
      _p_ctx(p_ctx),
      _fm_ctx(fm_ctx),
      _p_graph(p_graph),
      _shared(shared),
      _d_graph(&_p_graph),
      _d_gain_cache(_shared.gain_cache),
      _block_pq(_p_ctx.k),
      _stopping_policy(_fm_ctx.alpha) {
  _stopping_policy.init(_p_graph.n());
  for (const BlockID b : _p_graph.blocks()) {
    _node_pq.emplace_back(_p_ctx.n, _p_ctx.n, _shared.shared_pq_handles.data());
  }
}

EdgeWeight LocalizedFMRefiner::run_batch() {
  // Keep track of nodes that we don't want to unlock afterwards
  std::vector<NodeID> committed_moves;

  // Statistics for this batch only, to be merged into the global stats
  Stats stats;

  // Poll seed nodes from the border node arrays
  _shared.border_nodes
      .poll(_fm_ctx.num_seed_nodes, _id, [&](const NodeID seed_node) {
        insert_into_node_pq(_p_graph, _shared.gain_cache, seed_node);

        // Never unlock seed nodes, even if no move gets committed
        committed_moves.push_back(seed_node);
        IFSTATS(++stats.num_touched_nodes);
      });

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
      _shared.target_blocks[node] = block_to;
      if (_node_pq[block_from].peek_key() != _block_pq.key(block_from)) {
        _block_pq.change_priority(block_from, _node_pq[block_from].peek_key());
      }

      IFSTATS(++stats.num_recomputed_gains);
      continue;
    }

    // Otherwise, we can remove the node from the PQ
    _node_pq[block_from].pop();
    _shared.node_tracker.set(node, -1);
    IFSTATS(++stats.num_pq_pops);

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
        // DBG << "Worker " << _id << " committed local improvement with gain "
        //     << current_total_gain;

        // Update global graph and global gain cache
        for (const auto &[moved_node, moved_to] : _d_graph.delta()) {
          _shared.gain_cache.move(
              _p_graph, moved_node, _p_graph.block(moved_node), moved_to
          );
          _p_graph.set_block(moved_node, moved_to);
          committed_moves.push_back(moved_node);
          IFSTATS(++stats.num_committed_moves);
        }

        // Flush local delta
        _d_graph.clear();
        _d_gain_cache.clear();
        _stopping_policy.reset();

        best_total_gain = current_total_gain;
      }

      for (const auto &[e, v] : _p_graph.neighbors(node)) {
        if (_shared.node_tracker.owner(v) == _id) {
          KASSERT(_node_pq[_p_graph.block(v)].contains(v), "node not in PQ");
          update_after_move(v, node, block_from, block_to);
          IFSTATS(++stats.num_pq_updates);
        } else if (_shared.node_tracker.owner(v) == 0 && _shared.node_tracker.lock(v, _id)) {
          insert_into_node_pq(_d_graph, _d_gain_cache, v);
          IFSTATS(++stats.num_pq_inserts);

          touched_nodes.push_back(v);
          IFSTATS(++stats.num_touched_nodes);
        }
      }
    }
  }

  IFSTATS(stats.num_discarded_moves += _d_graph.delta().size());

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
    _shared.node_tracker.set(touched_node, 0);
  }

  // ... but keep nodes that we actually moved locked
  for (const NodeID moved_node : committed_moves) {
    _shared.node_tracker.set(moved_node, -1);
  }

  IFSTATS(_shared.stats.add(stats));
  return best_total_gain;
}

void LocalizedFMRefiner::update_after_move(
    const NodeID node,
    const NodeID moved_node,
    const BlockID moved_from,
    const BlockID moved_to
) {
  // KASSERT(_d_graph.block(node) == _p_graph.block(node));
  const BlockID old_block = _p_graph.block(node);
  const BlockID old_target_block = _shared.target_blocks[node];

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
      _shared.target_blocks[node] = new_target_block;
      _node_pq[old_block].change_priority(node, new_gain);
    }
  } else if (moved_from == old_target_block) {
    // old_target_block go worse, thus have to re-consider all other blocks
    const auto [new_target_block, new_gain] =
        best_gain(_d_graph, _d_gain_cache, node);
    _shared.target_blocks[node] = new_target_block;
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
      _shared.target_blocks[node] = moved_to;
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

template <typename PartitionedGraphType, typename GainCacheType>
void LocalizedFMRefiner::insert_into_node_pq(
    const PartitionedGraphType &p_graph,
    const GainCacheType &gain_cache,
    const NodeID u
) {
  const BlockID block_u = p_graph.block(u);
  const auto [block_to, gain] = best_gain(p_graph, gain_cache, u);
  KASSERT(!_node_pq[block_u].contains(u), "node already contained in PQ");
  _shared.target_blocks[u] = block_to;
  _node_pq[block_u].push(u, gain);
}

template <typename PartitionedGraphType, typename GainCacheType>
std::pair<BlockID, EdgeWeight> LocalizedFMRefiner::best_gain(
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
    const NodeWeight block_weight_gap = max_block_weight - target_block_weight;

    if (block_weight_gap < best_target_block_weight_gap &&
        block_weight_gap < 0) {
      continue;
    }

    const EdgeWeight gain = gain_cache.gain(u, block_u, block);
    if (gain > best_gain || (gain == best_gain &&
                             block_weight_gap > best_target_block_weight_gap)) {
      best_gain = gain;
      best_target_block = block;
      best_target_block_weight_gap = block_weight_gap;
    }
  }

  return {best_target_block, best_gain};
}
} // namespace kaminpar::shm

