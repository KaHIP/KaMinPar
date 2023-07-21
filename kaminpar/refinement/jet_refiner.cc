/*******************************************************************************
 * @file:   jet_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 * @brief:  Shared-memory JET refiner due to:
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 ******************************************************************************/
#include "kaminpar/refinement/jet_refiner.h"

#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/greedy_balancer.h"

#include "common/degree_buckets.h"
#include "common/logger.h"
#include "common/datastructures/noinit_vector.h"
#include "common/timer.h"

namespace kaminpar::shm {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(true);

struct Statistics {
  EdgeWeight pre_move_cut = 0;
  double pre_move_imbalance = 0.0;
  bool pre_move_feasible = false;

  EdgeWeight pre_rebalance_cut = 0;
  double pre_rebalance_imbalance = 0.0;
  bool pre_rebalance_feasible = false;

  EdgeWeight final_cut = 0;
  double final_imbalance = 0.0;
  bool final_feasible = false;

  parallel::Atomic<EdgeWeight> initial_pos_gain_sum = 0;
  parallel::Atomic<EdgeWeight> initial_gain_sum = 0;
  parallel::Atomic<NodeID> initial_num_pos_gain_moves = 0;
  parallel::Atomic<NodeID> initial_num_moves = 0;
  parallel::Atomic<EdgeWeight> prefiltered_gain_sum = 0;
  parallel::Atomic<NodeID> prefiltered_num_moves = 0;
  parallel::Atomic<EdgeWeight> filtered_expected_gain_sum = 0;
  parallel::Atomic<EdgeWeight> filtered_actual_gain_sum = 0;
  parallel::Atomic<NodeID> filtered_num_actual_pos_gain_moves = 0;
  parallel::Atomic<NodeID> filtered_num_actual_zero_gain_moves = 0;
  parallel::Atomic<NodeID> filtered_num_actual_neg_gain_moves = 0;
  parallel::Atomic<NodeID> filtered_num_moves = 0;

  std::array<parallel::Atomic<NodeID>, kNumberOfDegreeBuckets<NodeID>> filtered_num_moves_by_deg;
  std::array<parallel::Atomic<NodeID>, kNumberOfDegreeBuckets<NodeID>> rebalance_num_moves_by_deg;
  std::array<parallel::Atomic<NodeID>, kNumberOfDegreeBuckets<NodeID>> final_num_moves_by_deg;

  bool last_iteration_is_best = false;

  void print(const int iteration, const int max_iterations) {
    STATS << "JET:ITER(" << iteration << " of " << max_iterations << ") "
          << "pre_move_cut=" << pre_move_cut << " "
          << "pre_move_imbalance=" << pre_move_imbalance << " "
          << "pre_move_feasible=" << pre_move_feasible << " "
          << "pre_rebalance_cut=" << pre_rebalance_cut << " "
          << "pre_rebalance_imbalance=" << pre_rebalance_imbalance << " "
          << "pre_rebalance_feasible=" << pre_rebalance_feasible << " "
          << "final_cut=" << final_cut << " "
          << "final_imbalance=" << final_imbalance << " "
          << "final_feasible=" << final_feasible << " "
          << "initial_pos_gain_sum=" << initial_pos_gain_sum << " "
          << "initial_gain_sum=" << initial_gain_sum << " "
          << "initial_num_pos_gain_moves=" << initial_num_pos_gain_moves << " "
          << "initial_num_moves=" << initial_num_moves << " "
          << "prefiltered_gain_sum=" << prefiltered_gain_sum << " "
          << "prefiltered_num_moves=" << prefiltered_num_moves << " "
          << "filtered_expected_gain_sum=" << filtered_expected_gain_sum << " "
          << "filtered_actual_gain_sum=" << filtered_actual_gain_sum << " "
          << "filtered_num_actual_pos_gain_moves=" << filtered_num_actual_pos_gain_moves << " "
          << "filtered_num_actual_zero_gain_moves=" << filtered_num_actual_zero_gain_moves << " "
          << "filtered_num_actual_neg_gain_moves=" << filtered_num_actual_neg_gain_moves << " "
          << "last_iteration_is_best=" << last_iteration_is_best;
  }
};

JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

bool JetRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("JET");

  const NodeID min_size = p_ctx.k * _ctx.coarsening.contraction_limit;
  const NodeID cur_size = p_graph.n();
  const NodeID max_size = p_ctx.n;
  const double min_c = _ctx.refinement.jet.min_c;
  const double max_c = _ctx.refinement.jet.max_c;
  const double c = [&] {
    if (_ctx.refinement.jet.interpolate_c) {
      return min_c + (max_c - min_c) * (cur_size - min_size) / (max_size - min_size);
    } else {
      if (cur_size <= 2 * min_size) {
        return min_c;
      } else {
        return max_c;
      }
    }
  }();
  DBG << "Setting c=" << c;

  START_TIMER("Allocation");
  DenseGainCache gain_cache(p_graph.k(), p_graph.n());
  gain_cache.initialize(p_graph);

  NoinitVector<BlockID> next_partition(p_graph.n());
  p_graph.pfor_nodes([&](const NodeID u) { next_partition[u] = 0; });

  NoinitVector<std::uint8_t> lock(p_graph.n());
  p_graph.pfor_nodes([&](const NodeID u) { lock[u] = 0; });

  GreedyBalancer balancer(_ctx);
  balancer.initialize(p_graph);
  balancer.track_moves(&gain_cache);

  StaticArray<BlockID> best_partition(p_graph.n());
  p_graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
  EdgeWeight best_cut = metrics::edge_cut(p_graph);
  bool last_iteration_is_best = true;
  STOP_TIMER();

  for (int i = 0; i < _ctx.refinement.jet.num_iterations; ++i) {
    Statistics stats;

    const EdgeWeight initial_cut = metrics::edge_cut(p_graph);

    stats.pre_move_cut = IFSTATS(metrics::edge_cut(p_graph));
    stats.pre_move_imbalance = IFSTATS(metrics::imbalance(p_graph));
    stats.pre_move_feasible = IFSTATS(metrics::is_balanced(p_graph, p_ctx));

    TIMED_SCOPE("Find moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        const BlockID from = p_graph.block(u);

        if (lock[u] || !gain_cache.is_border_node(u, from)) {
          next_partition[u] = from;
          return;
        }

        EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
        BlockID best_block = from;

        for (const BlockID to : p_graph.blocks()) {
          if (to == from) {
            continue;
          }

          const EdgeWeight gain = gain_cache.gain(u, from, to);
          if (gain > best_gain) {
            best_gain = gain;
            best_block = to;
          }
        }

        IFSTATS(++stats.initial_num_moves);
        IFSTATS(stats.initial_gain_sum += best_gain);
        if (best_gain > 0) {
          IFSTATS(stats.initial_pos_gain_sum += best_gain);
        }

        if (-best_gain < std::floor(c * gain_cache.conn(u, from))) {
          next_partition[u] = best_block;

          IFSTATS(++stats.prefiltered_num_moves);
          IFSTATS(stats.prefiltered_gain_sum += best_gain);
        } else {
          next_partition[u] = from;
        }
      });
    };

    TIMED_SCOPE("Filter moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        lock[u] = 0;

        const BlockID from = p_graph.block(u);
        const BlockID to = next_partition[u];
        if (from == to) {
          return;
        }

        const EdgeWeight gain_u = gain_cache.gain(u, from, to);
        EdgeWeight gain = 0;

        for (const auto &[e, v] : p_graph.neighbors(u)) {
          const EdgeWeight weight = p_graph.edge_weight(e);

          const bool v_before_u = [&, v = v] {
            const BlockID from_v = p_graph.block(v);
            const BlockID to_v = next_partition[v];
            if (from_v != to_v) {
              const EdgeWeight gain_v = gain_cache.gain(v, from_v, to_v);
              return gain_v > gain_u || (gain_v == gain_u && v < u);
            }
            return false;
          }();
          const BlockID block_v = v_before_u ? next_partition[v] : p_graph.block(v);

          if (to == block_v) {
            gain += weight;
          } else if (from == block_v) {
            gain -= weight;
          }
        }

        if (gain > 0) {
          IFSTATS(++stats.filtered_num_moves);
          IFSTATS(stats.filtered_expected_gain_sum += gain);
          lock[u] = 1;
        }
      });
    };

    TIMED_SCOPE("Execute moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (lock[u]) {
          const BlockID from = p_graph.block(u);
          const BlockID to = next_partition[u];

          IFSTATS(
              stats.filtered_num_actual_pos_gain_moves += gain_cache.gain(u, from, to) > 0 ? 1 : 0
          );
          IFSTATS(
              stats.filtered_num_actual_zero_gain_moves += gain_cache.gain(u, from, to) == 0 ? 1 : 0
          );
          IFSTATS(
              stats.filtered_num_actual_neg_gain_moves += gain_cache.gain(u, from, to) < 0 ? 1 : 0
          );
          IFSTATS(stats.filtered_actual_gain_sum += gain_cache.gain(u, from, to));
          IFSTATS(stats.filtered_num_moves_by_deg[degree_bucket(p_graph.degree(u))]++);

          p_graph.set_block(u, to);
          gain_cache.move(p_graph, u, from, p_graph.block(u));
        }
      });
    };

    stats.pre_rebalance_cut = IFSTATS(metrics::edge_cut(p_graph));
    stats.pre_rebalance_imbalance = IFSTATS(metrics::imbalance(p_graph));
    stats.pre_rebalance_feasible = IFSTATS(metrics::is_balanced(p_graph, p_ctx));

    StaticArray<BlockID> imbalanced_partition(IFSTATS(p_graph.n()));
    IF_STATS {
      p_graph.pfor_nodes([&](const NodeID u) { imbalanced_partition[u] = p_graph.block(u); });
    }

    TIMED_SCOPE("Rebalance") {
      balancer.refine(p_graph, p_ctx);
    };

    IF_STATS {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (imbalanced_partition[u] != p_graph.block(u)) {
          IFSTATS(stats.rebalance_num_moves_by_deg[degree_bucket(p_graph.degree(u))]++);
        }
      });
    }

    TIMED_SCOPE("Update best partition") {
      const EdgeWeight current_cut = metrics::edge_cut(p_graph);
      if (current_cut <= best_cut) {
        p_graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
        best_cut = current_cut;
        last_iteration_is_best = true;
      } else {
        last_iteration_is_best = false;
      }
    };

    IFSTATS(stats.last_iteration_is_best = last_iteration_is_best);
    IFSTATS(stats.final_cut = metrics::edge_cut(p_graph));
    IFSTATS(stats.final_imbalance = metrics::imbalance(p_graph));
    IFSTATS(stats.final_feasible = metrics::is_feasible(p_graph, p_ctx));
    IFSTATS(stats.print(i, _ctx.refinement.jet.num_iterations));

    const EdgeWeight final_cut = metrics::edge_cut(p_graph);
    const double improvement = 1.0 * (initial_cut - final_cut) / initial_cut;
    if (1.0 - improvement > _ctx.refinement.jet.abortion_threshold) {
      break;
    }
  }

  TIMED_SCOPE("Rollback") {
    if (!last_iteration_is_best) {
      p_graph.pfor_nodes([&](const NodeID u) { p_graph.set_block(u, best_partition[u]); });
    }
  };

  return false;
}
} // namespace kaminpar::shm
