/*******************************************************************************
 * Parallel k-way FM refinement algorithm.
 *
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/fm/fm_refiner.h"

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/fm/fm_batch_stats.h"
#include "kaminpar-shm/refinement/fm/stopping_policies.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"

#ifdef KAMINPAR_EXPERIMENTAL
#include "kaminpar-shm/refinement/gains/hybrid_gain_cache.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"
#include "kaminpar-shm/refinement/gains/tracing_gain_cache.h"
#endif // KAMINPAR_EXPERIMENTAL

#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
namespace {
SET_DEBUG(false);
SET_STATISTICS(true);
} // namespace

std::unique_ptr<Refiner> create_fm_refiner(const Context &ctx) {
  switch (ctx.refinement.kway_fm.gain_cache_strategy) {
  case GainCacheStrategy::DENSE:
    return std::make_unique<FMRefiner<DenseGainCache<true>>>(ctx);

#ifdef KAMINPAR_EXPERIMENTAL
  case GainCacheStrategy::SPARSE:
    return std::make_unique<FMRefiner<SparseGainCache<true>>>(ctx);

  case GainCacheStrategy::ON_THE_FLY:
    return std::make_unique<FMRefiner<OnTheFlyGainCache<true>>>(ctx);

  case GainCacheStrategy::HYBRID:
    return std::make_unique<FMRefiner<HybridGainCache<true>>>(ctx);

  case GainCacheStrategy::TRACING:
    return std::make_unique<FMRefiner<TracingGainCache<SparseGainCache<true>>>>(ctx);
#endif

  default:
    LOG_ERROR << "invalid gain cache strategy: requires build with experimental features enabled";
    std::exit(1);
  }

  __builtin_unreachable();
}

namespace fm {
Stats &Stats::operator+=(const Stats &other) {
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

GlobalStats::GlobalStats() {
  next_iteration();
}

void GlobalStats::add(const Stats &stats) {
  iteration_stats.back() += stats;
}

void GlobalStats::next_iteration() {
  iteration_stats.emplace_back();
}

void GlobalStats::reset() {
  iteration_stats.clear();
  next_iteration();
}

void GlobalStats::print() {
  STATS << "FM Refiner:";
  for (std::size_t i = 0; i < iteration_stats.size(); ++i) {
    const Stats &stats = iteration_stats[i];
    if (stats.num_batches == 0) {
      continue;
    }

    STATS << "  * Iteration " << (i + 1) << ":";
    STATS << "    + # of batches: " << stats.num_batches;
    STATS << "    + # of touched nodes: " << stats.num_touched_nodes << " in total, "
          << 1.0 * stats.num_touched_nodes / stats.num_batches << " per batch";
    STATS << "    + # of moves: " << stats.num_committed_moves << " committed, "
          << stats.num_discarded_moves << " discarded (= "
          << 100.0 * stats.num_discarded_moves /
                 (stats.num_committed_moves + stats.num_discarded_moves)
          << "%)";
    STATS << "    + # of recomputed gains: " << stats.num_recomputed_gains;
    STATS << "    + # of PQ operations: " << stats.num_pq_inserts << " inserts, "
          << stats.num_pq_updates << " updates, " << stats.num_pq_pops << " pops";
  }
}
} // namespace fm

template <typename GainCache, typename DeltaPartitionedGraph>
FMRefiner<GainCache, DeltaPartitionedGraph>::FMRefiner(const Context &input_ctx)
    : _ctx(input_ctx),
      _fm_ctx(input_ctx.refinement.kway_fm),
      _shared(std::make_unique<fm::SharedData<GainCache>>(
          input_ctx, input_ctx.partition.n, input_ctx.partition.k
      )) {}

template <typename GainCache, typename DeltaPartitionedGraph>
FMRefiner<GainCache, DeltaPartitionedGraph>::~FMRefiner() = default;

template <typename GainCache, typename DeltaPartitionedGraph>
bool FMRefiner<GainCache, DeltaPartitionedGraph>::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("FM");

  START_TIMER("Initialize gain cache");
  _shared->gain_cache.initialize(p_graph);
  STOP_TIMER();

  const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
  EdgeWeight cut_before_current_iteration = initial_cut;
  EdgeWeight total_expected_gain = 0;

  // Create thread-local workers numbered 1..P
  using Worker = LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>;
  std::atomic<int> next_id = 0;
  tbb::enumerable_thread_specific<std::unique_ptr<Worker>> localized_fm_refiner_ets([&] {
    // It is important that worker IDs start at 1, otherwise the node
    // tracker won't work
    std::unique_ptr<Worker> localized_refiner =
        std::make_unique<Worker>(++next_id, p_ctx, _fm_ctx, p_graph, *_shared);

    // If we want to evaluate the successful batches, record moves that are applied to the
    // global graph
    IF_STATSC(_fm_ctx.dbg_compute_batch_stats) {
      localized_refiner->enable_move_recording();
    }

    return localized_refiner;
  });

  fm::BatchStatsComputator batch_stats(p_graph);

  for (int iteration = 0; iteration < _fm_ctx.num_iterations; ++iteration) {
    // Gains of the current iterations
    tbb::enumerable_thread_specific<EdgeWeight> expected_gain_ets;

    // Make sure that we work with correct gains // == 0, member variable
    KASSERT(
        _shared->gain_cache.validate(p_graph),
        "gain cache invalid before iteration " << iteration,
        assert::heavy
    );

    // Find current border nodes
    START_TIMER("Initialize border nodes");
    _shared->border_nodes.init(p_graph); // also resets the NodeTracker
    _shared->border_nodes.shuffle();
    STOP_TIMER();

    DBG << "Starting FM iteration " << iteration << " with " << _shared->border_nodes.size()
        << " border nodes and " << _ctx.parallel.num_threads << " worker threads";

    // Start one worker per thread
    if (p_graph.n() == _ctx.partition.n) {
      START_TIMER("Localized searches, fine level");
    } else {
      START_TIMER("Localized searches, coarse level");
    }
    tbb::parallel_for<int>(0, _ctx.parallel.num_threads, [&](int) {
      EdgeWeight &expected_gain = expected_gain_ets.local();
      LocalizedFMRefiner<GainCache, DeltaPartitionedGraph> &localized_refiner =
          *localized_fm_refiner_ets.local();

      // The workers attempt to extract seed nodes from the border nodes
      // that are still available, continuing this process until there are
      // no more border nodes
      while (_shared->border_nodes.has_more()) {
        const auto expected_batch_gain = localized_refiner.run_batch();
        expected_gain += expected_batch_gain;

        // Copies of the seed nodes and the moves are intentional: postpone actual stats computation
        // until after the FM iteration has finished
        IFSTATSC(
            _fm_ctx.dbg_compute_batch_stats && expected_batch_gain > 0,
            batch_stats.track(
                localized_refiner.last_batch_seed_nodes(), localized_refiner.last_batch_moves()
            )
        );
      }
    });
    STOP_TIMER();

    IFSTATSC(_fm_ctx.dbg_compute_batch_stats, batch_stats.next_iteration());

    const EdgeWeight expected_gain_of_this_iteration = expected_gain_ets.combine(std::plus{});
    total_expected_gain += expected_gain_of_this_iteration;

    const EdgeWeight current_cut =
        _fm_ctx.use_exact_abortion_threshold
            ? metrics::edge_cut(p_graph)
            : cut_before_current_iteration - expected_gain_of_this_iteration;

    const EdgeWeight abs_improvement_of_this_iteration = cut_before_current_iteration - current_cut;
    const double improvement_of_this_iteration =
        1.0 * abs_improvement_of_this_iteration / cut_before_current_iteration;
    if (1.0 - improvement_of_this_iteration > _fm_ctx.abortion_threshold) {
      break;
    }

    cut_before_current_iteration = current_cut;
    DBG << "Expected gain of iteration " << iteration << ": " << expected_gain_of_this_iteration
        << ", total expected gain so far: " << total_expected_gain;
    IFSTATS(_shared->stats.next_iteration());
  }

  IFSTATS(_shared->stats.print());
  IFSTATS(_shared->stats.reset());
  IFSTATS(_shared->gain_cache.print_statistics());
  IFSTATSC(_fm_ctx.dbg_compute_batch_stats, batch_stats.print());

  return false;
}

template <typename GainCache, typename DeltaPartitionedGraph>
LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::LocalizedFMRefiner(
    const int id,
    const PartitionContext &p_ctx,
    const KwayFMRefinementContext &fm_ctx,
    PartitionedGraph &p_graph,
    fm::SharedData<GainCache> &shared
)
    : _id(id),
      _p_ctx(p_ctx),
      _fm_ctx(fm_ctx),
      _p_graph(p_graph),
      _shared(shared),
      _d_graph(&_p_graph),
      _d_gain_cache(_shared.gain_cache, _d_graph),
      _block_pq(_p_graph.k()),
      _stopping_policy(_fm_ctx.alpha) {
  _stopping_policy.init(_p_graph.n());
  for (const BlockID b : _p_graph.blocks()) {
    _node_pqs.emplace_back(_p_graph.n(), _shared.shared_pq_handles.data());
  }
}

template <typename GainCache, typename DeltaPartitionedGraph>
void LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::enable_move_recording() {
  _record_applied_moves = true;
}

template <typename GainCache, typename DeltaPartitionedGraph>
const std::vector<fm::AppliedMove> &
LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::last_batch_moves() {
  return _applied_moves;
}

template <typename GainCache, typename DeltaPartitionedGraph>
const std::vector<NodeID> &
LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::last_batch_seed_nodes() {
  return _seed_nodes;
}

template <typename GainCache, typename DeltaPartitionedGraph>
EdgeWeight LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::run_batch() {
  using fm::NodeTracker;

  _seed_nodes.clear();
  _applied_moves.clear();

  // Statistics for this batch only, to be merged into the global stats
  fm::Stats stats;
  IFSTATS(++stats.num_batches);

  // Poll seed nodes from the border node arrays
  _shared.border_nodes.poll(_fm_ctx.num_seed_nodes, _id, [&](const NodeID seed_node) {
    insert_into_node_pq(_p_graph, _shared.gain_cache, seed_node);
    _seed_nodes.push_back(seed_node);
    IFSTATS(++stats.num_touched_nodes);
  });

  // Keep track of the current (expected) gain to decide when to accept a
  // delta partition
  EdgeWeight current_total_gain = 0;
  EdgeWeight best_total_gain = 0;

  while (update_block_pq() && !_stopping_policy.should_stop()) {
    const BlockID block_from = _block_pq.peek_id();
    KASSERT(block_from < _p_graph.k());

    const NodeID node = _node_pqs[block_from].peek_id();
    KASSERT(node < _p_graph.n());

    const EdgeWeight expected_gain = _node_pqs[block_from].peek_key();
    const auto [block_to, actual_gain] = best_gain(_d_graph, _d_gain_cache, node);

    // If the gain got worse, reject the move and try again
    if (actual_gain < expected_gain) {
      _node_pqs[block_from].change_priority(node, actual_gain);
      _shared.target_blocks[node] = block_to;
      if (_node_pqs[block_from].peek_key() != _block_pq.key(block_from)) {
        _block_pq.change_priority(block_from, _node_pqs[block_from].peek_key());
      }

      IFSTATS(++stats.num_recomputed_gains);
      continue;
    }

    // Otherwise, we can remove the node from the PQ
    _node_pqs[block_from].pop();
    _shared.node_tracker.set(node, NodeTracker::MOVED_LOCALLY);
    IFSTATS(++stats.num_pq_pops);

    // Skip the move if there is no viable target block
    if (block_to == block_from) {
      continue;
    }

    // Accept the move if the target block does not get overloaded
    const NodeWeight node_weight = _p_graph.node_weight(node);
    if (_d_graph.block_weight(block_to) + node_weight <= _p_ctx.block_weights.max(block_to)) {
      current_total_gain += actual_gain;

      // If we found a new local minimum, apply the moves to the global
      // partition
      if (current_total_gain > best_total_gain) {
        _p_graph.set_block(node, block_to);
        _shared.gain_cache.move(_p_graph, node, block_from, block_to);
        _shared.node_tracker.set(node, NodeTracker::MOVED_GLOBALLY);
        IFSTATS(++stats.num_committed_moves);

        _d_graph.for_each([&](const NodeID moved_node, const BlockID moved_to) {
          const BlockID moved_from = _p_graph.block(moved_node);

          // The order of the moves in the delta graph is not necessarily correct (depending on
          // whether the delta graph stores the moves in a vector of a hash table).
          // Thus, users of the _applied_moves vector may only depend on the order of moves that
          // found an improvement.
          if (_record_applied_moves) {
            _applied_moves.push_back(fm::AppliedMove{
                .node = moved_node,
                .from = moved_from,
                .improvement = false,
            });
          }

          _shared.gain_cache.move(_p_graph, moved_node, moved_from, moved_to);
          _shared.node_tracker.set(moved_node, NodeTracker::MOVED_GLOBALLY);
          _p_graph.set_block(moved_node, moved_to);
          IFSTATS(++stats.num_committed_moves);
        });

        if (_record_applied_moves) {
          _applied_moves.push_back(fm::AppliedMove{
              .node = node,
              .from = block_from,
              .improvement = true,
          });
        }

        // Flush local delta
        _d_graph.clear();
        _d_gain_cache.clear();
        _stopping_policy.reset();

        best_total_gain = current_total_gain;
      } else {
        // Perform local move
        _d_graph.set_block(node, block_to);
        _d_gain_cache.move(_d_graph, node, block_from, block_to);
        _stopping_policy.update(actual_gain);
      }

      for (const auto &[e, v] : _p_graph.neighbors(node)) {
        const int owner = _shared.node_tracker.owner(v);
        if (owner == _id) {
          KASSERT(_node_pqs[_p_graph.block(v)].contains(v), "owned node not in PQ");
          update_after_move(v, node, block_from, block_to);
          IFSTATS(++stats.num_pq_updates);
        } else if (owner == NodeTracker::UNLOCKED && _shared.node_tracker.lock(v, _id)) {
          insert_into_node_pq(_d_graph, _d_gain_cache, v);
          IFSTATS(++stats.num_pq_inserts);

          _touched_nodes.push_back(v);
          IFSTATS(++stats.num_touched_nodes);
        }
      }
    }
  }

  // Flush local state for the nex tround
  for (auto &node_pq : _node_pqs) {
    node_pq.clear();
  }

  auto unlock_touched_node = [&](const NodeID node) {
    const int owner = _shared.node_tracker.owner(node);
    if (owner == NodeTracker::MOVED_LOCALLY) {
      if (_fm_ctx.unlock_locally_moved_nodes) {
        _shared.node_tracker.set(node, NodeTracker::UNLOCKED);
      } else {
        _shared.node_tracker.set(node, NodeTracker::MOVED_GLOBALLY);
      }
    } else if (owner == _id) {
      _shared.node_tracker.set(node, NodeTracker::UNLOCKED);
    }
  };

  // If we do not wish to unlock seed nodes, mark them as globally moved == locked for good
  for (const NodeID &seed_node : _seed_nodes) {
    if (!_fm_ctx.unlock_seed_nodes) {
      _shared.node_tracker.set(seed_node, NodeTracker::MOVED_GLOBALLY);
    } else {
      unlock_touched_node(seed_node);
    }
  }

  // Unlock all nodes that were touched but not moved, or nodes that were only moved in the
  // thread-local delta graph
  IFSTATS(stats.num_discarded_moves += _d_graph.size());
  for (const NodeID touched_node : _touched_nodes) {
    unlock_touched_node(touched_node);
  }

  _block_pq.clear();
  _d_graph.clear();
  _d_gain_cache.clear();
  _stopping_policy.reset();
  _touched_nodes.clear();

  IFSTATS(_shared.stats.add(stats));
  return best_total_gain;
}

template <typename GainCache, typename DeltaPartitionedGraph>
inline void LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::update_after_move(
    const NodeID node, const NodeID moved_node, const BlockID moved_from, const BlockID moved_to
) {
  const BlockID old_block = _p_graph.block(node);
  const BlockID old_target_block = _shared.target_blocks[node];

  if (moved_to == old_target_block) {
    // In this case, old_target_block got even better
    // We only need to consider other blocks if old_target_block is full now
    if (_d_graph.block_weight(old_target_block) + _d_graph.node_weight(node) <=
        _p_ctx.block_weights.max(old_target_block)) {
      _node_pqs[old_block].change_priority(
          node, _d_gain_cache.gain(node, old_block, old_target_block)
      );
    } else {
      const auto [new_target_block, new_gain] = best_gain(_d_graph, _d_gain_cache, node);
      _shared.target_blocks[node] = new_target_block;
      _node_pqs[old_block].change_priority(node, new_gain);
    }
  } else if (moved_from == old_target_block) {
    // old_target_block go worse, thus have to re-consider all other blocks
    const auto [new_target_block, new_gain] = best_gain(_d_graph, _d_gain_cache, node);
    _shared.target_blocks[node] = new_target_block;
    _node_pqs[old_block].change_priority(node, new_gain);
  } else if (moved_to == old_block) {
    KASSERT(moved_from != old_target_block);
    // Since we did not move from old_target_block, this block is still the
    // best and we can still move to that block
    _node_pqs[old_block].change_priority(
        node, _d_gain_cache.gain(node, old_block, old_target_block)
    );
  } else {
    // old_target_block OR moved_to is best
    const auto [gain_old_target_block, gain_moved_to] =
        _d_gain_cache.gain(node, old_block, {old_target_block, moved_to});

    if (gain_moved_to > gain_old_target_block &&
        _d_graph.block_weight(moved_to) + _d_graph.node_weight(node) <=
            _p_ctx.block_weights.max(moved_to)) {
      _shared.target_blocks[node] = moved_to;
      _node_pqs[old_block].change_priority(node, gain_moved_to);
    } else {
      _node_pqs[old_block].change_priority(node, gain_old_target_block);
    }
  }
}

template <typename GainCache, typename DeltaPartitionedGraph>
inline bool LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::update_block_pq() {
  bool have_more_nodes = false;
  for (const BlockID block : _p_graph.blocks()) {
    if (!_node_pqs[block].empty()) {
      const EdgeWeight gain = _node_pqs[block].peek_key();
      _block_pq.push_or_change_priority(block, gain);
      have_more_nodes = true;
    } else if (_block_pq.contains(block)) {
      _block_pq.remove(block);
    }
  }
  return have_more_nodes;
}

template <typename GainCache, typename DeltaPartitionedGraph>
template <typename GainCacheType, typename PartitionedGraphType>
inline void LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::insert_into_node_pq(
    const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, const NodeID u
) {
  const BlockID block_u = p_graph.block(u);
  const auto [block_to, gain] = best_gain(p_graph, gain_cache, u);
  KASSERT(block_u < _node_pqs.size(), "block_u out of bounds");
  KASSERT(!_node_pqs[block_u].contains(u), "node already contained in PQ");
  _shared.target_blocks[u] = block_to;
  _node_pqs[block_u].push(u, gain);
}

template <typename GainCache, typename DeltaPartitionedGraph>
template <typename GainCacheType, typename PartitionedGraphType>
inline std::pair<BlockID, EdgeWeight>
LocalizedFMRefiner<GainCache, DeltaPartitionedGraph>::best_gain(
    const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, const NodeID u
) {
  const BlockID from = p_graph.block(u);
  const NodeWeight weight = p_graph.node_weight(u);

  // Since we use max heaps, it is OK to insert this value into the PQ
  EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
  BlockID best_target_block = from;
  NodeWeight best_target_block_weight_gap =
      _p_ctx.block_weights.max(from) - p_graph.block_weight(from);

  gain_cache.gains(u, from, [&](const BlockID to, auto &&compute_gain) {
    const NodeWeight target_block_weight = p_graph.block_weight(to) + weight;
    const NodeWeight max_block_weight = _p_ctx.block_weights.max(to);
    const NodeWeight block_weight_gap = max_block_weight - target_block_weight;
    if (block_weight_gap < std::min<EdgeWeight>(best_target_block_weight_gap, 0)) {
      return;
    } // loop could not be vectorized

    const EdgeWeight gain = compute_gain();
    if (gain > best_gain ||
        (gain == best_gain && block_weight_gap > best_target_block_weight_gap)) {
      best_gain = gain; // SLOW: hoisting getelementptr
      best_target_block = to;
      best_target_block_weight_gap = block_weight_gap;
    }
  });

  const EdgeWeight actual_best_gain = [&] {
    if (best_target_block == from) {
      return std::numeric_limits<EdgeWeight>::min();
    } else {
      if constexpr (GainCacheType::kIteratesExactGains) {
        return best_gain;
      } else {
        return gain_cache.gain(u, from, best_target_block);
      }
    }
  }();

  return {best_target_block, actual_best_gain};
}
} // namespace kaminpar::shm
