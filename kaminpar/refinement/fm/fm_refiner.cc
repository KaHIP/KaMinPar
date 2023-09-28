/*******************************************************************************
 * Parallel k-way FM refinement algorithm.
 *
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#include "kaminpar/refinement/fm/fm_refiner.h"

#include <cmath>
#include <queue>
#include <set>
#include <unordered_map>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/fm/stopping_policies.h"
#include "kaminpar/refinement/gains/on_the_fly_gain_cache.h"

#include "common/datastructures/marker.h"
#include "common/logger.h"
#include "common/parallel/atomic.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::shm {
namespace {
SET_DEBUG(false);
SET_STATISTICS(true);
} // namespace

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

void GlobalStats::summarize() {
  LOG_STATS << "FM Refinement:";
  for (std::size_t i = 0; i < iteration_stats.size(); ++i) {
    const Stats &stats = iteration_stats[i];
    if (stats.num_batches == 0) {
      continue;
    }

    LOG_STATS << "  * Iteration " << (i + 1) << ":";
    LOG_STATS << "    + Number of batches: " << stats.num_batches;
    LOG_STATS << "    + Number of touched nodes: " << stats.num_touched_nodes << " in total, "
              << 1.0 * stats.num_touched_nodes / stats.num_batches << " per batch";
    LOG_STATS << "    + Number of moves: " << stats.num_committed_moves << " committed, "
              << stats.num_discarded_moves << " discarded (= "
              << 100.0 * stats.num_discarded_moves /
                     (stats.num_committed_moves + stats.num_discarded_moves)
              << "%)";
    LOG_STATS << "    + Number of recomputed gains: " << stats.num_recomputed_gains;
    LOG_STATS << "    + Number of PQ operations: " << stats.num_pq_inserts << " inserts, "
              << stats.num_pq_updates << " updates, " << stats.num_pq_pops << " pops";
  }
}
void GlobalBatchStats::next_iteration(std::vector<BatchStats> stats) {
  iteration_stats.push_back(std::move(stats));
}

void GlobalBatchStats::reset() {
  iteration_stats.clear();
}

void GlobalBatchStats::summarize() {
  LOG_STATS << "Batches: [STATS:FM:BATCHES]";
  for (std::size_t i = 0; i < iteration_stats.size(); ++i) {
    if (!iteration_stats[i].empty()) {
      LOG_STATS << "  * Iteration " << (i + 1) << ":";
      summarize_iteration(i, iteration_stats[i]);
    }
  }
}

void GlobalBatchStats::summarize_iteration(
    const std::size_t iteration, const std::vector<BatchStats> &stats
) {
  const NodeID max_distance =
      std::max_element(stats.begin(), stats.end(), [&](const auto &lhs, const auto &rhs) {
        return lhs.max_distance < rhs.max_distance;
      })->max_distance;

  std::vector<NodeID> total_size_by_distance(max_distance + 1);
  std::vector<EdgeWeight> total_gain_by_distance(max_distance + 1);
  for (NodeID distance = 0; distance <= max_distance; ++distance) { // <=
    for (const auto &batch_stats : stats) {
      if (distance < batch_stats.size_by_distance.size()) {
        KASSERT(distance < batch_stats.gain_by_distance.size());
        total_size_by_distance[distance] += batch_stats.size_by_distance[distance];
        total_gain_by_distance[distance] += batch_stats.gain_by_distance[distance];
      }
    }
  }

  LOG_STATS << "    - Max distance: " << max_distance << " [STATS:FM:BATCHES:" << iteration << "]";
  std::stringstream size_ss, gain_ss;
  size_ss << "      + Size by distance: " << total_size_by_distance[0];
  gain_ss << "      + Gain by distance: " << total_gain_by_distance[0];

  for (NodeID distance = 1; distance <= max_distance; ++distance) { // <=
    size_ss << "," << total_size_by_distance[distance];
    gain_ss << "," << total_gain_by_distance[distance];
  }
  LOG_STATS << size_ss.str() << " [STATS:FM:BATCHES:" << iteration << "]";
  LOG_STATS << gain_ss.str() << " [STATS:FM:BATCHES:" << iteration << "]";
}
} // namespace fm

template <typename GainCache>
FMRefiner<GainCache>::FMRefiner(const Context &input_ctx)
    : _ctx(input_ctx),
      _fm_ctx(input_ctx.refinement.kway_fm),
      _shared(
          std::make_unique<fm::SharedData<GainCache>>(input_ctx.partition.n, input_ctx.partition.k)
      ) {}

template <typename GainCache> FMRefiner<GainCache>::~FMRefiner() = default;

template <typename GainCache>
bool FMRefiner<GainCache>::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("FM");

  START_TIMER("Initialize gain cache");
  _shared->gain_cache.initialize(p_graph);
  STOP_TIMER();

  const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
  EdgeWeight cut_before_current_iteration = initial_cut;
  EdgeWeight total_expected_gain = 0;

  // Create thread-local workers numbered 1..P
  std::atomic<int> next_id = 0;
  tbb::enumerable_thread_specific<LocalizedFMRefiner<GainCache>> localized_fm_refiner_ets([&] {
    // It is important that worker IDs start at 1, otherwise the node
    // tracker won't work
    LocalizedFMRefiner localized_refiner(++next_id, p_ctx, _fm_ctx, p_graph, *_shared);

    // If we want to evaluate the successful batches, record moves that are applied to the global
    // graph
    IF_STATSC(_fm_ctx.dbg_compute_batch_size_statistics) {
      localized_refiner.enable_move_recording();
    }

    return localized_refiner;
  });

  for (int iteration = 0; iteration < _fm_ctx.num_iterations; ++iteration) {
    // Gains of the current iterations
    tbb::enumerable_thread_specific<EdgeWeight> expected_gain_ets;

    // Make sure that we work with correct gains
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

    // If we want to evaluate the batches, record all batches and their changes to the partition
    Batches dbg_changelog;

    // Start one worker per thread
    START_TIMER("Localized searches");
    tbb::parallel_for<int>(0, _ctx.parallel.num_threads, [&](int) {
      EdgeWeight &expected_gain = expected_gain_ets.local();
      LocalizedFMRefiner<GainCache> &localized_refiner = localized_fm_refiner_ets.local();

      // The workers attempt to extract seed nodes from the border nodes
      // that are still available, continuing this process until there are
      // no more border nodes
      while (_shared->border_nodes.has_more()) {
        const auto expected_batch_gain = localized_refiner.run_batch();
        expected_gain += expected_batch_gain;
        IF_STATSC(_fm_ctx.dbg_compute_batch_size_statistics && expected_batch_gain > 0) {
          SeedNodesVec seed_nodes_cpy(localized_refiner.last_batch_seed_nodes());
          MovesVec moves_cpy(localized_refiner.last_batch_moves());
          dbg_changelog.emplace_back(std::move(seed_nodes_cpy), std::move(moves_cpy));
        }
      }
    });
    STOP_TIMER();

    IF_STATSC(_fm_ctx.dbg_compute_batch_size_statistics) {
      std::vector<fm::BatchStats> batch_stats =
          dbg_compute_batch_stats(p_graph, std::move(dbg_changelog));
      _shared->batch_stats.next_iteration(std::move(batch_stats));
    }

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

  IF_STATS {
    _shared->stats.summarize();
    _shared->stats.reset();
  }
  IF_STATSC(_fm_ctx.dbg_compute_batch_size_statistics) {
    _shared->batch_stats.summarize();
    _shared->batch_stats.reset();
  }

  return false;
}

template <typename GainCache>
std::vector<fm::BatchStats> FMRefiner<GainCache>::dbg_compute_batch_stats(
    const PartitionedGraph &next_p_graph, Batches next_batches
) const {
  // Rollback the partition to *before* any moves of the batch were applied
  // prev_batches will now contain the *target* block for all nodes instead of their previous block
  auto [prev_p_graph, prev_batches] = dbg_build_prev_p_graph(next_p_graph, std::move(next_batches));

  std::vector<std::vector<NodeID>> batch_distances(prev_batches.size());
  tbb::parallel_for<std::size_t>(
      0,
      prev_batches.size(),
      [&, &prev_p_graph = prev_p_graph, &prev_batches = prev_batches](std::size_t i) {
        const auto &[seeds, moves] = prev_batches[i];
        if (!moves.empty()) {
          batch_distances[i] = dbg_compute_batch_distances(prev_p_graph.graph(), seeds, moves);
        }
      }
  );

  // In the recorded sequence, re-apply batches batch-by-batch to measure their effect on partition
  // quality
  std::vector<fm::BatchStats> batch_stats;
  for (std::size_t i = 0; i < prev_batches.size(); ++i) {
    const auto &[seeds, moves] = prev_batches[i];
    const auto &distances = batch_distances[i];

    if (!moves.empty()) {
      batch_stats.push_back(
          dbg_compute_single_batch_stats_in_sequence(prev_p_graph, seeds, moves, distances)
      );
    } else {
      batch_stats.emplace_back();
    }
  }

  // If everything went right, we should now have the same partition as next_partition
  KASSERT(metrics::edge_cut(prev_p_graph) == metrics::edge_cut(next_p_graph), "", assert::always);
  KASSERT(metrics::imbalance(prev_p_graph) == metrics::imbalance(next_p_graph), "", assert::always);

  return batch_stats;
}

// Computes the partition *before* any moves of the given batches where applied to it
// Changes the batches to store the blocks to which the nodes where moved to
template <typename GainCache>
auto FMRefiner<GainCache>::dbg_build_prev_p_graph(const PartitionedGraph &p_graph, Batches batches)
    const -> std::pair<PartitionedGraph, FMRefiner::Batches> {
  StaticArray<BlockID> prev_partition(p_graph.n());
  auto &next_partition = p_graph.partition();
  std::copy(next_partition.begin(), next_partition.end(), prev_partition.begin());

  // Rollback partition to before the moves in the batches where applied
  // Update the batches to store the "new" from block
  for (auto &[seeds, moves] : batches) {
    for (auto &move : moves) {
      std::swap(prev_partition[move.node], move.from);
    }
  }

  return {
      PartitionedGraph(p_graph.graph(), p_graph.k(), std::move(prev_partition)),
      std::move(batches),
  };
}

// Computes the statistics for a single batch
// The given partition should reflect all batches that came before this one, but none of the ones
// that will come afterwards
// This function also applies the moves of the current batch to the given partition
template <typename GainCache>
fm::BatchStats FMRefiner<GainCache>::dbg_compute_single_batch_stats_in_sequence(
    PartitionedGraph &p_graph,
    const std::vector<NodeID> &seeds,
    const std::vector<fm::AppliedMove> &moves,
    const std::vector<NodeID> &distances
) const {
  KASSERT(!seeds.empty());
  KASSERT(!moves.empty());
  fm::BatchStats stats;

  stats.size = moves.size();
  stats.max_distance = *std::max_element(distances.begin(), distances.end());
  stats.size_by_distance.resize(stats.max_distance + 1);
  stats.gain_by_distance.resize(stats.max_distance + 1);

  NodeID cur_distance = 0;

  EdgeWeight gain_for_next_improvement = 0;
  NodeID size_for_next_improvement = 0;

  for (std::size_t i = 0; i < moves.size(); ++i) {
    const auto &[u, block, improvement] = moves[i];

    // Compute the gain of the move
    EdgeWeight int_degree = 0;
    EdgeWeight ext_degree = 0;
    for (const auto &[e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) == p_graph.block(u)) {
        int_degree += p_graph.edge_weight(e);
      } else if (p_graph.block(v) == block) {
        ext_degree += p_graph.edge_weight(e);
      }
    }

    KASSERT(i < distances.size());
    cur_distance = std::max(cur_distance, distances[i]);

    gain_for_next_improvement += ext_degree - int_degree;
    size_for_next_improvement += 1;

    if (improvement) {
      stats.gain_by_distance[cur_distance] += gain_for_next_improvement;
      stats.size_by_distance[cur_distance] += size_for_next_improvement;
      gain_for_next_improvement = 0;
      size_for_next_improvement = 0;
    }

    p_graph.set_block(u, block);
  }

  return stats;
}

template <typename GainCache>
std::vector<NodeID> FMRefiner<GainCache>::dbg_compute_batch_distances(
    const Graph &graph, const std::vector<NodeID> &seeds, const std::vector<fm::AppliedMove> &moves
) const {
  // Keeps track of moved nodes that we yet have to discover
  std::unordered_map<NodeID, std::size_t> searched;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    searched[moves[i].node] = i;
  }

  // Keep track of nodes that we have already discovered
  std::set<NodeID> visited;

  // Current frontier of the BFS
  std::queue<NodeID> frontier;
  for (const NodeID &seed : seeds) {
    frontier.push(seed);
    visited.insert(seed);
  }

  NodeID current_distance = 0;
  NodeID current_layer_size = frontier.size();
  std::vector<NodeID> distances(moves.size());

  while (!searched.empty()) {
    KASSERT(!frontier.empty());

    if (current_layer_size == 0) {
      ++current_distance;
      current_layer_size = frontier.size();
    }

    const NodeID u = frontier.front();
    frontier.pop();
    --current_layer_size;

    // If the node was moved, record its distance from any seed node
    if (auto it = searched.find(u); it != searched.end()) {
      distances[it->second] = current_distance;
      searched.erase(it);
    }

    // Expand search to its neighbors
    for (const auto &[e, v] : graph.neighbors(u)) {
      if (visited.count(v) == 0) {
        visited.insert(v);
        frontier.push(v);
      }
    }
  }

  return distances;
}

template <typename GainCache>
LocalizedFMRefiner<GainCache>::LocalizedFMRefiner(
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

template <typename GainCache> void LocalizedFMRefiner<GainCache>::enable_move_recording() {
  _record_applied_moves = true;
}

template <typename GainCache>
const std::vector<fm::AppliedMove> &LocalizedFMRefiner<GainCache>::last_batch_moves() {
  return _applied_moves;
}

template <typename GainCache>
const std::vector<NodeID> &LocalizedFMRefiner<GainCache>::last_batch_seed_nodes() {
  return _seed_nodes;
}

template <typename GainCache> EdgeWeight LocalizedFMRefiner<GainCache>::run_batch() {
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

        for (const auto &[moved_node, moved_to] : _d_graph.delta()) {
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
        }

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

  // If we do not wish to unlock seed nodes, mark them as globally moved == locked for good
  if (!_fm_ctx.unlock_seed_nodes) {
    for (const NodeID &seed_node : _seed_nodes) {
      _shared.node_tracker.set(seed_node, NodeTracker::MOVED_GLOBALLY);
    }
  } else {
    for (const NodeID seed_node : _seed_nodes) {
      const int owner = _shared.node_tracker.owner(seed_node);
      if (owner == _id || owner == NodeTracker::MOVED_LOCALLY) {
        _shared.node_tracker.set(seed_node, NodeTracker::UNLOCKED);
      }
    }
  }

  // Unlock all nodes that were touched but not moved, or nodes that were only moved in the
  // thread-local delta graph
  IFSTATS(stats.num_discarded_moves += _d_graph.delta().size());
  for (const NodeID touched_node : _touched_nodes) {
    const int owner = _shared.node_tracker.owner(touched_node);
    if (owner == _id || owner == NodeTracker::MOVED_LOCALLY) {
      _shared.node_tracker.set(touched_node, NodeTracker::UNLOCKED);
    }
  }

  _block_pq.clear();
  _d_graph.clear();
  _d_gain_cache.clear();
  _stopping_policy.reset();
  _touched_nodes.clear();

  IFSTATS(_shared.stats.add(stats));
  return best_total_gain;
}

template <typename GainCache>
void LocalizedFMRefiner<GainCache>::update_after_move(
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
    const EdgeWeight gain_old_target_block = _d_gain_cache.gain(node, old_block, old_target_block);
    const EdgeWeight gain_moved_to = _d_gain_cache.gain(node, old_block, moved_to);

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

template <typename GainCache> bool LocalizedFMRefiner<GainCache>::update_block_pq() {
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

template <typename GainCache>
template <typename PartitionedGraphType, typename GainCacheType>
void LocalizedFMRefiner<GainCache>::insert_into_node_pq(
    const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, const NodeID u
) {
  const BlockID block_u = p_graph.block(u);
  const auto [block_to, gain] = best_gain(p_graph, gain_cache, u);
  KASSERT(block_u < _node_pqs.size(), "block_u out of bounds");
  KASSERT(!_node_pqs[block_u].contains(u), "node already contained in PQ");
  _shared.target_blocks[u] = block_to;
  _node_pqs[block_u].push(u, gain);
}

template <typename GainCache>
template <typename PartitionedGraphType, typename GainCacheType>
std::pair<BlockID, EdgeWeight> LocalizedFMRefiner<GainCache>::best_gain(
    const PartitionedGraphType &p_graph, const GainCacheType &gain_cache, const NodeID u
) {
  const BlockID block_u = p_graph.block(u);
  const NodeWeight weight_u = p_graph.node_weight(u);

  // Since we use max heaps, it is OK to insert this value into the PQ
  EdgeWeight best_conn = std::numeric_limits<EdgeWeight>::min();
  BlockID best_target_block = block_u;
  NodeWeight best_target_block_weight_gap =
      _p_ctx.block_weights.max(block_u) - p_graph.block_weight(block_u);

  gain_cache.gains(
      u,
      block_u,
      [&](const BlockID block) {
        const NodeWeight target_block_weight = p_graph.block_weight(block) + weight_u;
        const NodeWeight max_block_weight = _p_ctx.block_weights.max(block);
        const NodeWeight block_weight_gap = max_block_weight - target_block_weight;
        return block_weight_gap >= best_target_block_weight_gap || block_weight_gap >= 0;
      },
      [&](const BlockID block, const EdgeWeight conn) {
        const NodeWeight target_block_weight = p_graph.block_weight(block) + weight_u;
        const NodeWeight max_block_weight = _p_ctx.block_weights.max(block);
        const NodeWeight block_weight_gap = max_block_weight - target_block_weight;

        if (conn > best_conn ||
            (conn == best_conn && block_weight_gap > best_target_block_weight_gap)) {
          best_conn = conn;
          best_target_block = block;
          best_target_block_weight_gap = block_weight_gap;
        }
      }
  );

  const EdgeWeight best_gain = [&] {
    if (best_target_block == block_u) {
      return std::numeric_limits<EdgeWeight>::min();
    } else {
      if constexpr (GainCacheType::kIteratesExactGains) {
        return best_conn;
      } else {
        return gain_cache.gain(u, block_u, best_target_block);
      }
    }
  }();

  return {best_target_block, best_gain};
}

namespace fm {
template class SharedData<DenseGainCache<>>;
template class SharedData<OnTheFlyGainCache<>>;
} // namespace fm

template class FMRefiner<DenseGainCache<>>;
template class FMRefiner<OnTheFlyGainCache<>>;

template class LocalizedFMRefiner<DenseGainCache<>>;
template class LocalizedFMRefiner<OnTheFlyGainCache<>>;
} // namespace kaminpar::shm
