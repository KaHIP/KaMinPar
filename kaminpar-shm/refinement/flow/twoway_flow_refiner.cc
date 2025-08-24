/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter_algorithm.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/hyper_flow_cutter.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/util/lazy_vector.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

class FlowRefiner {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  using Move = FlowCutterAlgorithm::Move;
  using Result = FlowCutterAlgorithm::Result;

public:
  FlowRefiner(
      const PartitionContext &p_ctx,
      const TwowayFlowRefinementContext &f_ctx,
      const bool run_sequentially,
      const QuotientGraph &q_graph,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      const TimePoint &start_time
  )
      : _p_graph(p_graph),
        _run_sequentially(run_sequentially),
        _border_region_constructor(p_ctx, f_ctx.construction, q_graph, p_graph, graph),
        _flow_network_constructor(p_graph, graph) {
#ifdef KAMINPAR_WHFC_FOUND
    if (f_ctx.flow_cutter.use_whfc) {
      _flow_cutter_algorithm =
          std::make_unique<HyperFlowCutter>(p_ctx, f_ctx.flow_cutter, run_sequentially);
    } else {
      _flow_cutter_algorithm =
          std::make_unique<FlowCutter>(p_ctx, f_ctx.flow_cutter, run_sequentially, p_graph);
    }
#else
    if (f_ctx.flow_cutter.use_whfc) {
      LOG_WARNING << "WHFC requested but not available; using built-in FlowCutter as fallback.";
    }

    _flow_cutter_algorithm =
        std::make_unique<FlowCutter>(p_ctx, f_ctx.flow_cutter, run_sequentially, p_graph);
#endif

    _flow_cutter_algorithm->set_time_limit(f_ctx.time_limit, start_time);
  }

  Result refine(const BlockID block1, const BlockID block2) {
    KASSERT(block1 != block2, "Only different block pairs can be refined");
    SCOPED_TIMER("Refine Block Pair");

    const BlockWeight block_weight1 = _p_graph.block_weight(block1);
    const BlockWeight block_weight2 = _p_graph.block_weight(block2);

    const BorderRegion &border_region =
        _border_region_constructor.construct(block1, block2, block_weight1, block_weight2);

    const FlowNetwork flow_network = _flow_network_constructor.construct_flow_network(
        border_region, block_weight1, block_weight2, _run_sequentially
    );

    return _flow_cutter_algorithm->compute_cut(border_region, flow_network);
  }

private:
  const PartitionedCSRGraph &_p_graph;
  const bool _run_sequentially;

  BorderRegionConstructor _border_region_constructor;
  FlowNetworkConstructor _flow_network_constructor;
  std::unique_ptr<FlowCutterAlgorithm> _flow_cutter_algorithm;
};

namespace {

struct BlockPairSchedulerStatistics {
  std::size_t num_searches;
  std::size_t num_local_improvements;
  std::size_t num_global_improvements;
  std::size_t num_move_conflicts;
  std::size_t num_imbalance_conflicts;
  std::size_t num_failed_imbalance_resolutions;
  std::size_t num_successful_imbalance_resolutions;
  double min_imbalance;
  double max_imbalance;
  double total_imbalance;

  void reset() {
    num_searches = 0;
    num_local_improvements = 0;
    num_global_improvements = 0;
    num_move_conflicts = 0;
    num_imbalance_conflicts = 0;
    num_failed_imbalance_resolutions = 0;
    num_successful_imbalance_resolutions = 0;
    min_imbalance = std::numeric_limits<double>::max();
    max_imbalance = std::numeric_limits<double>::min();
    total_imbalance = 0.0;
  }

  void print() const {
    LOG_STATS << "Two-Way Flow Refiner:";
    LOG_STATS << "*  # num searches: " << num_searches;
    LOG_STATS << "*  # num local improvements: " << num_local_improvements;
    LOG_STATS << "*  # num global improvements: " << num_global_improvements;
    LOG_STATS << "*  # num move conflicts: " << num_move_conflicts;
    LOG_STATS << "*  # num imbalance conflicts: " << num_imbalance_conflicts;
    LOG_STATS << "*  # num failed imbalance resolutions: " << num_failed_imbalance_resolutions;
    LOG_STATS << "*  # num successful imbalance resolutions: "
              << num_successful_imbalance_resolutions;
    LOG_STATS << "*  # min / average / max imbalance: "
              << (num_imbalance_conflicts ? min_imbalance : 0) << " / "
              << (num_imbalance_conflicts ? total_imbalance / num_imbalance_conflicts : 0) << " / "
              << (num_imbalance_conflicts ? max_imbalance : 0);
  }
};

} // namespace

class SequentialActiveBlockScheduler {
  SET_DEBUG(true);
  SET_STATISTICS(true);

  using Clock = FlowRefiner::Clock;
  using TimePoint = FlowRefiner::TimePoint;
  using Move = FlowRefiner::Move;
  using Result = FlowRefiner::Result;

public:
  SequentialActiveBlockScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;

    if (_active_blocks.size() < _p_graph->k()) {
      _active_blocks.resize(_p_graph->k(), static_array::noinit);
    }

    // Since the timers have a significant running time overhead, we disable them usually.
    IF_NOT_DBG DISABLE_TIMERS();
    IF_STATS _stats.reset();

    const TimePoint start_time = Clock::now();
    QuotientGraph quotient_graph(p_graph);

    constexpr bool kActivateAllBlockPairs = true;
    activate_blocks(quotient_graph, kActivateAllBlockPairs);

    FlowRefiner refiner(
        p_ctx, _f_ctx, _f_ctx.run_sequentially, quotient_graph, p_graph, graph, start_time
    );

    std::size_t num_round = 0;
    bool found_improvement = false;

    double prev_imbalance = metrics::imbalance(*_p_graph);
    EdgeWeight prev_cut_value = quotient_graph.total_cut_weight();

    while (prev_cut_value > 0) {
      num_round += 1;
      DBG << "Starting round " << num_round;
      IF_STATS _stats.num_searches += _active_block_pairs.size();

      EdgeWeight cut_value = prev_cut_value;
      for (const auto &[block1, block2] : _active_block_pairs) {
        DBG << "Scheduling block pair " << block1 << " and " << block2;

        const Result result = refiner.refine(block1, block2);

        if (result.time_limit_exceeded) {
          LOG_WARNING << "Time limit exceeded during flow refinement";
          num_round = _f_ctx.max_num_rounds;
          break;
        }

        const EdgeWeight new_cut_value = cut_value - result.gain;
        DBG << "Found balanced cut for block pair " << block1 << " and " << block2 << " with gain "
            << result.gain << " (" << cut_value << " -> " << new_cut_value << ")";

        if (result.gain > 0 || (result.gain == 0 && result.improved_balance)) {
          apply_moves(result.moves);

          KASSERT(
              metrics::edge_cut_seq(p_graph) == new_cut_value,
              "Computed an invalid gain",
              assert::heavy
          );

          KASSERT(
              metrics::is_balanced(p_graph, p_ctx),
              "Computed an imbalanced move sequence",
              assert::heavy
          );

          IF_STATS _stats.num_local_improvements += 1;
          IF_STATS _stats.num_global_improvements += 1;

          cut_value = new_cut_value;

          quotient_graph.add_cut_edges(_new_cut_edges);
          quotient_graph.add_gain(block1, block2, result.gain);

          _active_blocks[block1] = true;
          _active_blocks[block2] = true;
        }
      }

      const EdgeWeight round_gain = prev_cut_value - cut_value;
      const double imbalance = metrics::imbalance(*_p_graph);
      found_improvement |= round_gain > 0 || imbalance < prev_imbalance;

      const double relative_improvement = round_gain / static_cast<double>(prev_cut_value);
      DBG << "Finished round with a relative improvement of " << relative_improvement;

      if (num_round == _f_ctx.max_num_rounds ||
          relative_improvement < _f_ctx.min_round_improvement_factor) {
        break;
      }

      quotient_graph.reconstruct();
      activate_blocks(quotient_graph);

      prev_cut_value = cut_value;
      prev_imbalance = imbalance;
    }

    IF_NOT_DBG ENABLE_TIMERS();
    IF_STATS _stats.print();

    return found_improvement;
  }

private:
  void activate_blocks(const QuotientGraph &quotient_graph, const bool activate_all = false) {
    SCOPED_TIMER("Activate Blocks");

    _active_block_pairs.clear();
    for (BlockID block2 = 1, k = _p_graph->k(); block2 < k; ++block2) {
      for (BlockID block1 = 0; block1 < block2; ++block1) {
        if (quotient_graph.has_quotient_edge(block1, block2) &&
            (activate_all || _active_blocks[block1] || _active_blocks[block2])) {
          _active_block_pairs.emplace_back(block1, block2);
        }
      }
    }

    std::sort(
        _active_block_pairs.begin(),
        _active_block_pairs.end(),
        [&](const auto &pair1, const auto &pair2) {
          const QuotientGraph::Edge &edge1 = quotient_graph.edge(pair1.first, pair1.second);
          const QuotientGraph::Edge &edge2 = quotient_graph.edge(pair2.first, pair2.second);
          return edge1.total_gain > edge2.total_gain ||
                 (edge1.total_gain == edge2.total_gain && (edge1.cut_weight > edge2.cut_weight));
        }
    );

    std::fill_n(_active_blocks.begin(), _p_graph->k(), false);
  }

  void apply_moves(std::span<const Move> moves) {
    SCOPED_TIMER("Apply Moves");

    _new_cut_edges.clear();
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
          _new_cut_edges.emplace_back(u, v);
        }
      });
    }
  }

private:
  const TwowayFlowRefinementContext &_f_ctx;
  BlockPairSchedulerStatistics _stats;

  PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;

  StaticArray<bool> _active_blocks;
  ScalableVector<std::pair<BlockID, BlockID>> _active_block_pairs;

  ScalableVector<QuotientGraph::GraphEdge> _new_cut_edges;
};

class ParallelActiveBlockScheduler {
  SET_DEBUG(true);
  SET_STATISTICS(true);

  using Clock = FlowRefiner::Clock;
  using TimePoint = FlowRefiner::TimePoint;
  using Move = FlowRefiner::Move;

  enum class MoveResult {
    IMBALANCE_CONFLICT,
    NEGATIVE_GAIN,
    SUCCESS,
  };

  struct MoveAttempt {
    MoveResult kind;
    double imbalance;
    EdgeWeight cut_value;
    EdgeWeight gain;

    MoveAttempt(MoveResult kind, double imbalance)
        : kind(kind),
          imbalance(imbalance),
          cut_value(0),
          gain(0) {}

    MoveAttempt(MoveResult kind, EdgeWeight cut_value, EdgeWeight gain)
        : kind(kind),
          imbalance(0.0),
          cut_value(cut_value),
          gain(gain) {}
  };

public:
  ParallelActiveBlockScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;
    _p_ctx = &p_ctx;

    if (_active_blocks.size() < p_graph.k()) {
      _active_blocks.resize(p_graph.k(), static_array::noinit);
    }
    if (_block_weight_delta.size() < p_graph.k()) {
      _block_weight_delta.resize(p_graph.k(), static_array::noinit);
    }

    // Since timers are not multi-threaded, we disable them during parallel refinement.
    DISABLE_TIMERS();
    IF_STATS _stats.reset();

    const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
    const std::size_t max_num_quotient_graph_edges = p_graph.k() * (p_graph.k() - 1) / 2;
    const std::size_t num_parallel_searches = std::min(
        std::min(num_threads, max_num_quotient_graph_edges),
        std::max<std::size_t>(1, _f_ctx.scheduler.parallel_searches_multiplier * p_graph.k())
    );

    const TimePoint start_time = Clock::now();
    QuotientGraph quotient_graph(p_graph);

    constexpr bool kActivateAllBlockPairs = true;
    activate_blocks(quotient_graph, kActivateAllBlockPairs);

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

    double prev_imbalance = metrics::imbalance(*_p_graph);
    EdgeWeight prev_cut_value = quotient_graph.total_cut_weight();

    while (prev_cut_value > 0) {
      num_round += 1;
      DBG << "Starting round " << num_round;
      IF_STATS _stats.num_searches += _active_block_pairs.size();

      const std::size_t num_searches = std::min(num_parallel_searches, _active_block_pairs.size());
      std::size_t cur_block_pair = 0;

      EdgeWeight cut_value = prev_cut_value;
      tbb::parallel_for<std::size_t>(0, num_searches, [&](const std::size_t refiner_id) {
        FlowRefiner &refiner = refiners[refiner_id];
        ScalableVector<QuotientGraph::GraphEdge> &new_cut_edges = new_cut_edges_ets[refiner_id];

        while (true) {
          const std::size_t block_pair = __atomic_fetch_add(&cur_block_pair, 1, __ATOMIC_RELAXED);
          if (block_pair >= _active_block_pairs.size()) {
            break;
          }

          const auto [block1, block2] = _active_block_pairs[block_pair];
          DBG << "Scheduling block pair " << block1 << " and " << block2;

          auto [time_limit_exceeded, gain, improve_balance, moves] = refiner.refine(block1, block2);

          if (time_limit_exceeded) {
            if (tbb::task::current_context()->cancel_group_execution()) {
              LOG_WARNING << "Time limit exceeded during flow refinement";
              num_round = _f_ctx.max_num_rounds;
            }

            return;
          }

          if (gain > 0 || (gain == 0 && improve_balance)) {
            IF_STATS _stats.num_local_improvements += 1;

            const MoveAttempt result = commit_moves_if_feasible(
                cut_value, block1, block2, moves, quotient_graph, new_cut_edges
            );

            if (result.kind == MoveResult::IMBALANCE_CONFLICT) {
              IF_STATS _stats.num_imbalance_conflicts += 1;
              IF_STATS _stats.min_imbalance = std::min(_stats.min_imbalance, result.imbalance);
              IF_STATS _stats.max_imbalance = std::max(_stats.max_imbalance, result.imbalance);
              IF_STATS _stats.total_imbalance += result.imbalance;
              DBG << "Block pair " << block1 << " and " << block2 << " has an imbalance conflict";

              if (_f_ctx.scheduler.reschedule_imbalance_conflicts) {
                _rescheduled_block_pairs.emplace_back(block1, block2);
              }

              return;
            }

            DBG << "Found balanced cut for block pair " << block1 << " and " << block2
                << " with gain " << result.gain << " (" << (result.cut_value + result.gain)
                << " -> " << result.cut_value << ")";

            if (result.kind == MoveResult::SUCCESS) {
              IF_STATS _stats.num_global_improvements += 1;

              quotient_graph.add_cut_edges(new_cut_edges);
            }
          }
        }
      });

      const EdgeWeight round_gain = prev_cut_value - cut_value;
      const double imbalance = metrics::imbalance(*_p_graph);
      found_improvement |= round_gain > 0 || imbalance < prev_imbalance;

      const double relative_improvement = round_gain / static_cast<double>(prev_cut_value);
      DBG << "Finished round with a relative improvement of " << relative_improvement;

      if (num_round == _f_ctx.max_num_rounds ||
          relative_improvement < _f_ctx.min_round_improvement_factor) {
        break;
      }

      quotient_graph.reconstruct();
      activate_blocks(quotient_graph);

      prev_cut_value = cut_value;
      prev_imbalance = imbalance;
    }

    ENABLE_TIMERS();
    IF_STATS _stats.print();

    return found_improvement;
  }

private:
  void activate_blocks(const QuotientGraph &quotient_graph, const bool activate_all = false) {
    SCOPED_TIMER("Activate Blocks");

    _active_block_pairs.clear();
    for (BlockID block2 = 1, k = _p_graph->k(); block2 < k; ++block2) {
      for (BlockID block1 = 0; block1 < block2; ++block1) {
        if (quotient_graph.has_quotient_edge(block1, block2) &&
            (activate_all || _active_blocks[block1] || _active_blocks[block2])) {
          _active_block_pairs.emplace_back(block1, block2);
        }
      }
    }

    for (const auto &[block1, block2] : _rescheduled_block_pairs) {
      if (!quotient_graph.has_quotient_edge(block1, block2) || activate_all ||
          _active_blocks[block1] || _active_blocks[block2]) {
        continue;
      }

      _active_block_pairs.emplace_back(block1, block2);
    }
    _rescheduled_block_pairs.clear();

    std::sort(
        _active_block_pairs.begin(),
        _active_block_pairs.end(),
        [&](const auto &pair1, const auto &pair2) {
          const QuotientGraph::Edge &edge1 = quotient_graph.edge(pair1.first, pair1.second);
          const QuotientGraph::Edge &edge2 = quotient_graph.edge(pair2.first, pair2.second);
          return edge1.total_gain > edge2.total_gain ||
                 (edge1.total_gain == edge2.total_gain && (edge1.cut_weight > edge2.cut_weight));
        }
    );

    std::fill_n(_active_blocks.begin(), _p_graph->k(), false);
  }

  MoveAttempt commit_moves_if_feasible(
      EdgeWeight &cut_value,
      const BlockID block1,
      const BlockID block2,
      const std::span<Move> moves,
      QuotientGraph &quotient_graph,
      ScalableVector<QuotientGraph::GraphEdge> &new_cut_edges
  ) {
    const std::unique_lock lock(_apply_moves_mutex);

    const auto [balanced, imbalance] = is_feasible_move_sequence(moves);
    if (!balanced) {
      return MoveAttempt(MoveResult::IMBALANCE_CONFLICT, imbalance);
    }

    const EdgeWeight actual_gain = apply_moves(moves, new_cut_edges);
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
    quotient_graph.add_gain(block1, block2, actual_gain);

    _active_blocks[block1] = true;
    _active_blocks[block2] = true;

    return MoveAttempt(MoveResult::SUCCESS, new_cut_value, actual_gain);
  }

  std::pair<bool, double> is_feasible_move_sequence(std::span<Move> moves) {
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

    const double perfect_block_weight =
        std::ceil(1.0 * _graph->total_node_weight() / _p_graph->k());

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

  EdgeWeight apply_moves(
      std::span<const Move> moves, ScalableVector<QuotientGraph::GraphEdge> &new_cut_edges
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

  void revert_moves(std::span<const Move> moves) {
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

private:
  const TwowayFlowRefinementContext &_f_ctx;
  BlockPairSchedulerStatistics _stats;

  PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;

  StaticArray<bool> _active_blocks;
  ScalableVector<std::pair<BlockID, BlockID>> _active_block_pairs;
  ScalableVector<std::pair<BlockID, BlockID>> _rescheduled_block_pairs;

  std::mutex _apply_moves_mutex;
  StaticArray<BlockWeight> _block_weight_delta;
};

TwowayFlowRefiner::TwowayFlowRefiner(
    const ParallelContext &par_ctx, const TwowayFlowRefinementContext &f_ctx
)
    : _par_ctx(par_ctx),
      _f_ctx(f_ctx),
      _sequential_scheduler(std::make_unique<SequentialActiveBlockScheduler>(f_ctx)),
      _parallel_scheduler(std::make_unique<ParallelActiveBlockScheduler>(f_ctx)) {}

TwowayFlowRefiner::~TwowayFlowRefiner() = default;

std::string TwowayFlowRefiner::name() const {
  return "Two-Way Flow Refinement";
}

void TwowayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool TwowayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) {
        // The bipartition refiner works with PartitionedCSRGraph instead of PartitionedGraph.
        // Intead of copying the partition, use a view on it to access the partition.
        PartitionedCSRGraph p_csr_graph = p_graph.csr_view();

        const bool found_improvement = refine(p_csr_graph, csr_graph, p_ctx);
        return found_improvement;
      },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the two-way flow refiner.";
        return false;
      }
  );
}

bool TwowayFlowRefiner::refine(
    PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("Two-Way Flow Refinement");
  SCOPED_HEAP_PROFILER("Two-Way Flow Refinement");

  if (_f_ctx.scheduler.parallel_scheduling && _par_ctx.num_threads > 1 && p_ctx.k > 2) {
    return _parallel_scheduler->refine(p_graph, graph, p_ctx);
  } else {
    return _sequential_scheduler->refine(p_graph, graph, p_ctx);
  }
}

} // namespace kaminpar::shm
