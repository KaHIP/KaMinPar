#include "kaminpar-shm/refinement/flow/scheduler/parallel_active_block_scheduler.h"

#include <algorithm>

#include <tbb/parallel_for.h>
#include <tbb/task.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/matching_based_active_block_scheduler.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/single_round_active_block_scheduler.h"
#include "kaminpar-shm/refinement/flow/util/lazy_vector.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

ParallelActiveBlockScheduler::ParallelActiveBlockScheduler(const TwowayFlowRefinementContext &f_ctx)
    : _f_ctx(f_ctx) {
  if (_f_ctx.scheduler.deterministic) {
    _active_block_scheduling = std::make_unique<MatchingBasedActiveBlockScheduling>();
  } else {
    _active_block_scheduling = std::make_unique<SingleRoundActiveBlockScheduling>();
  }
}

bool ParallelActiveBlockScheduler::refine(
    PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  _p_graph = &p_graph;
  _graph = &graph;
  _p_ctx = &p_ctx;

  if (_active_blocks.size() < p_graph.k()) {
    _active_blocks.resize(p_graph.k(), static_array::noinit);
  }
  if (_block_weight_delta.size() < p_graph.k()) {
    _block_weight_delta.resize(p_graph.k(), static_array::noinit);
  }
  std::fill_n(_active_blocks.begin(), p_graph.k(), true);

  // Since timers are not multi-threaded, we disable them during parallel refinement.
  DISABLE_TIMERS();
  IF_STATS _stats.reset();

  const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
  const std::size_t max_num_quotient_graph_edges = p_graph.k() * (p_graph.k() - 1) / 2;
  const std::size_t num_parallel_searches = std::min(
      std::min(num_threads, max_num_quotient_graph_edges),
      std::max<std::size_t>(1, _f_ctx.scheduler.parallel_search_multiplier * p_graph.k())
  );

  const TimePoint start_time = Clock::now();
  QuotientGraph quotient_graph(p_graph);

  const bool run_sequentially = _f_ctx.run_sequentially || num_threads == num_parallel_searches;
  LazyVector<FlowRefiner> refiners(
      [&] {
        return FlowRefiner(
            p_ctx, _f_ctx, run_sequentially, quotient_graph, p_graph, graph, start_time
        );
      },
      num_parallel_searches
  );
  LazyVector<ScalableVector<QuotientGraph::GraphEdge>> new_cut_edges_ets(num_parallel_searches);

  std::size_t num_round = 0;
  bool found_improvement = false;

  double prev_imbalance = metrics::imbalance(p_graph);
  EdgeWeight prev_cut_value = quotient_graph.total_cut_weight();

  while (prev_cut_value > 0) {
    num_round += 1;
    DBG << "Starting round " << num_round;

    const Scheduling scheduling =
        _active_block_scheduling->compute_scheduling(quotient_graph, _active_blocks);
    std::fill_n(_active_blocks.begin(), p_graph.k(), false);

    EdgeWeight cut_value = prev_cut_value;
    for (std::size_t subround = 0; subround < scheduling.size(); ++subround) {
      const auto &active_block_pairs = scheduling[subround];
      IF_STATS _stats.num_searches += active_block_pairs.size();

      const std::size_t num_searches = std::min(num_parallel_searches, active_block_pairs.size());
      std::size_t cur_block_pair = 0;

      tbb::parallel_for<std::size_t>(0, num_searches, [&](const std::size_t refiner_id) {
        FlowRefiner &refiner = refiners[refiner_id];
        ScalableVector<QuotientGraph::GraphEdge> &new_cut_edges = new_cut_edges_ets[refiner_id];

        while (true) {
          const std::size_t block_pair = __atomic_fetch_add(&cur_block_pair, 1, __ATOMIC_RELAXED);
          if (block_pair >= active_block_pairs.size()) {
            break;
          }

          const auto [block1, block2] = active_block_pairs[block_pair];
          DBG << "Scheduling block pair " << block1 << " and " << block2;

          const FlowRefiner::Result flow_result = refiner.refine(block1, block2);

          if (flow_result.time_limit_exceeded) {
            if (tbb::task::current_context()->cancel_group_execution()) {
              LOG_WARNING << "Time limit exceeded during flow refinement";
              num_round = _f_ctx.max_num_rounds;
            }

            return;
          }

          if (flow_result.gain > 0 || (flow_result.gain == 0 && flow_result.improved_balance)) {
            IF_STATS _stats.num_local_improvements += 1;

            if (_f_ctx.scheduler.deterministic) {
              IF_STATS _stats.num_global_improvements += 1;

              commit_moves(
                  cut_value,
                  flow_result.gain,
                  block1,
                  block2,
                  flow_result.moves,
                  quotient_graph,
                  new_cut_edges
              );

              DBG << "Found balanced cut for block pair " << block1 << " and " << block2
                  << " with gain " << flow_result.gain << " (" << (cut_value + flow_result.gain)
                  << " -> " << cut_value << ")";
            } else {
              const MoveAttempt result = commit_moves_if_feasible(
                  cut_value, block1, block2, flow_result.moves, new_cut_edges
              );

              if (result.kind == MoveResult::IMBALANCE_CONFLICT) {
                IF_STATS _stats.num_imbalance_conflicts += 1;
                IF_STATS _stats.min_imbalance = std::min(_stats.min_imbalance, result.imbalance);
                IF_STATS _stats.max_imbalance = std::max(_stats.max_imbalance, result.imbalance);
                IF_STATS _stats.total_imbalance += result.imbalance;

                DBG << "Block pair " << block1 << " and " << block2 << " has an imbalance conflict";
                continue;
              }

              DBG << "Found balanced cut for block pair " << block1 << " and " << block2
                  << " with gain " << result.gain << " (" << (result.cut_value + result.gain)
                  << " -> " << result.cut_value << ")";

              if (result.kind == MoveResult::SUCCESS) {
                IF_STATS _stats.num_global_improvements += 1;

                quotient_graph.add_cut_edges(new_cut_edges);
                quotient_graph.add_gain(block1, block2, result.gain);
              }
            }
          }
        }
      });
    }

    const EdgeWeight round_gain = prev_cut_value - cut_value;
    const double imbalance = metrics::imbalance(*_p_graph);
    found_improvement |= round_gain > 0 || imbalance < prev_imbalance;

    const double relative_improvement = round_gain / static_cast<double>(prev_cut_value);
    DBG << "Finished round with a relative improvement of " << relative_improvement
        << " and imbalance of " << imbalance;

    if (num_round == _f_ctx.max_num_rounds ||
        relative_improvement < _f_ctx.min_round_improvement_factor) {
      break;
    }

    quotient_graph.reconstruct();

    prev_cut_value = cut_value;
    prev_imbalance = imbalance;
  }

  ENABLE_TIMERS();
  IF_STATS _stats.print();

  return found_improvement;
}

void ParallelActiveBlockScheduler::commit_moves(
    EdgeWeight &cut_value,
    const EdgeWeight gain,
    const BlockID block1,
    const BlockID block2,
    const std::span<const Move> moves,
    QuotientGraph &quotient_graph,
    QuotientCutEdges &new_cut_edges
) {
  apply_moves(moves, new_cut_edges);

  __atomic_fetch_sub(&cut_value, gain, __ATOMIC_RELAXED);

  quotient_graph.add_cut_edges(new_cut_edges);
  quotient_graph.add_gain(block1, block2, gain);

  _active_blocks[block1] = true;
  _active_blocks[block2] = true;
}

void ParallelActiveBlockScheduler::apply_moves(
    std::span<const Move> moves, QuotientCutEdges &new_cut_edges
) {
  new_cut_edges.clear();
  for (const Move &move : moves) {
    KASSERT(
        _p_graph->block(move.node) == move.old_block,
        "Move sequence contains invalid old block ids",
        assert::heavy
    );
    KASSERT(
        move.old_block != move.new_block,
        "Move sequence contains moves where node is already in target block",
        assert::heavy
    );

    const NodeID u = move.node;
    const BlockID new_block = move.new_block;
    _p_graph->set_block(u, new_block);

    const BlockID old_block = move.old_block;
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      const BlockID v_block = _p_graph->block(v);

      if (v_block == old_block) {
        new_cut_edges.emplace_back(u, v);
      }
    });
  }
}

ParallelActiveBlockScheduler::MoveAttempt ParallelActiveBlockScheduler::commit_moves_if_feasible(
    EdgeWeight &cut_value,
    const BlockID block1,
    const BlockID block2,
    const std::span<Move> moves,
    QuotientCutEdges &new_cut_edges
) {
  const std::unique_lock lock(_apply_moves_mutex);

  const auto [balanced, imbalance] = is_feasible_move_sequence(moves);
  if (!balanced) {
    return MoveAttempt(MoveResult::IMBALANCE_CONFLICT, imbalance);
  }

  const EdgeWeight actual_gain = atomic_apply_moves(moves, new_cut_edges);
  const EdgeWeight new_cut_value = cut_value - actual_gain;

  KASSERT(
      metrics::edge_cut_seq(*_p_graph) == new_cut_value,
      "Computed an invalid new cut value",
      assert::heavy
  );
  KASSERT(
      metrics::is_balanced(*_p_graph, *_p_ctx),
      "Computed an imbalanced move sequence",
      assert::heavy
  );

  if (actual_gain < 0) {
    revert_moves(moves);
    return MoveAttempt(MoveResult::NEGATIVE_GAIN, new_cut_value, actual_gain);
  }

  cut_value = new_cut_value;

  _active_blocks[block1] = true;
  _active_blocks[block2] = true;

  return MoveAttempt(MoveResult::SUCCESS, new_cut_value, actual_gain);
}

std::pair<bool, double>
ParallelActiveBlockScheduler::is_feasible_move_sequence(std::span<Move> moves) {
  std::fill_n(_block_weight_delta.begin(), _p_graph->k(), 0);

  for (Move &move : moves) {
    const NodeID u = move.node;
    const BlockID old_block = move.old_block;
    const BlockID new_block = move.new_block;

    KASSERT(
        move.old_block != move.new_block,
        "Move sequence contains moves where node is already in target block",
        assert::heavy
    );

    // Remove all nodes from the move sequence that are not in their expected block.
    // Use the old block variable to mark the move as such, which is used during reverting.
    const BlockID invalid_block_conflict = _p_graph->block(u) != old_block;
    if (invalid_block_conflict) {
      IF_STATS _stats.num_move_conflicts += 1;

      move.old_block = kInvalidBlockID;
      continue;
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    _block_weight_delta[old_block] -= u_weight;
    _block_weight_delta[new_block] += u_weight;
  }

  const double perfect_block_weight = std::ceil(1.0 * _graph->total_node_weight() / _p_graph->k());

  bool balanced = true;
  double max_imbalance = 0.0;
  for (const BlockID b : _p_graph->blocks()) {
    const BlockWeight b_weight = _p_graph->block_weight(b) + _block_weight_delta[b];

    if (b_weight > _p_ctx->max_block_weight(b)) {
      balanced = false;
      max_imbalance = std::max(max_imbalance, b_weight / perfect_block_weight - 1.0);
    }
  }

  return {balanced, max_imbalance};
}

EdgeWeight ParallelActiveBlockScheduler::atomic_apply_moves(
    const std::span<const Move> moves, QuotientCutEdges &new_cut_edges
) {
  EdgeWeight gain = 0;

  new_cut_edges.clear();
  for (const Move &move : moves) {
    const BlockID old_block = move.old_block;

    // If the node was not in its expected block, it should not be moved.
    const BlockID invalid_block_conflict = old_block == kInvalidBlockID;
    if (invalid_block_conflict) {
      continue;
    }

    const NodeID u = move.node;
    const BlockID new_block = move.new_block;
    _p_graph->set_block(u, new_block);

    EdgeWeight from_connection = 0;
    EdgeWeight to_connection = 0;
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      const BlockID v_block = _p_graph->block(v);
      to_connection += (v_block == new_block) ? w : 0;

      if (v_block == old_block) {
        from_connection += w;
        new_cut_edges.emplace_back(u, v);
      }
    });
    gain += to_connection - from_connection;
  }

  return gain;
}

void ParallelActiveBlockScheduler::revert_moves(std::span<const Move> moves) {
  for (const Move &move : moves) {
    const BlockID old_block = move.old_block;

    // If the node was not in its expected block, it has not been moved.
    // Thus, the move must not be reverted.
    const BlockID invalid_block_conflict = old_block == kInvalidBlockID;
    if (invalid_block_conflict) {
      continue;
    }

    const NodeID u = move.node;
    _p_graph->set_block(u, old_block);
  }
}

} // namespace kaminpar::shm
