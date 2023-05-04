#include "kaminpar/refinement/jet_refiner.h"

#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/greedy_balancer.h"

#include "common/logger.h"
#include "common/noinit_vector.h"
#include "common/timer.h"

namespace kaminpar::shm {
SET_DEBUG(true);
SET_STATISTICS(true);

JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

bool JetRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("JET");

  const NodeID min_size = p_ctx.k * _ctx.coarsening.contraction_limit;
  const NodeID cur_size = p_graph.n();
  const NodeID max_size = p_ctx.n;
  const double min_c = _ctx.refinement.jet.min_c;
  const double max_c = _ctx.refinement.jet.max_c;
  const double c = [&] {
    if (_ctx.refinement.jet.interpolate_c) {
      return min_c +
             (max_c - min_c) * (cur_size - min_size) / (max_size - min_size);
    } else {
      if (cur_size <= 2 * min_size) {
        return min_c;
      } else {
        return max_c;
      }
    }
  }();
  DBG << "Set c=" << c;

  TIMED_SCOPE("Statistics") {
    const EdgeWeight initial_cut = IFDBG(metrics::edge_cut(p_graph));
    const double initial_balance = IFDBG(metrics::imbalance(p_graph));
    const bool initial_feasible = IFDBG(metrics::is_feasible(p_graph, p_ctx));
    DBG << "Initial cut=" << initial_cut << ", imbalance=" << initial_balance
        << ", feasible=" << initial_feasible;
  };

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
  p_graph.pfor_nodes([&](const NodeID u) {
    best_partition[u] = p_graph.block(u);
  });
  EdgeWeight best_cut = metrics::edge_cut(p_graph);
  bool last_iteration_is_best = true;
  STOP_TIMER();

  for (int i = 0; i < _ctx.refinement.jet.num_iterations; ++i) {
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

    parallel::Atomic<NodeID> filtered_num_moves_deg01perc = 0;
    parallel::Atomic<NodeID> filtered_num_moves_deg1perc = 0;
    parallel::Atomic<NodeID> filtered_num_moves_deg50k = 0;

    parallel::Atomic<NodeID> rebalance_num_moves = 0;
    parallel::Atomic<NodeID> rebalance_num_moves_deg01perc = 0;
    parallel::Atomic<NodeID> rebalance_num_moves_deg1perc = 0;
    parallel::Atomic<NodeID> rebalance_num_moves_deg50k = 0;

    parallel::Atomic<NodeID> final_num_doublemoved = 0;
    parallel::Atomic<NodeID> final_num_doublemoved_deg01perc = 0;
    parallel::Atomic<NodeID> final_num_doublemoved_deg1perc = 0;
    parallel::Atomic<NodeID> final_num_doublemoved_deg50k = 0;

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

        IFDBG(++initial_num_moves);
        IFDBG(initial_gain_sum += best_gain);
        if (best_gain > 0) {
          IFDBG(initial_pos_gain_sum += best_gain);
        }

        if (-best_gain < std::floor(c * gain_cache.conn(u, from))) {
          next_partition[u] = best_block;

          IFDBG(++prefiltered_num_moves);
          IFDBG(prefiltered_gain_sum += best_gain);
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
          const BlockID block_v =
              v_before_u ? next_partition[v] : p_graph.block(v);

          if (to == block_v) {
            gain += weight;
          } else if (from == block_v) {
            gain -= weight;
          }
        }

        if (gain > 0) {
          IFDBG(++filtered_num_moves);
          IFDBG(filtered_expected_gain_sum += gain);
          lock[u] = 1;
        }
      });
    };

    TIMED_SCOPE("Execute moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (lock[u]) {
          const BlockID from = p_graph.block(u);
          const BlockID to = next_partition[u];

          const EdgeWeight actual_gain = IFDBG(gain_cache.gain(u, from, to));
          IFDBG(filtered_num_actual_pos_gain_moves += actual_gain > 0 ? 1 : 0);
          IFDBG(
              filtered_num_actual_zero_gain_moves += actual_gain == 0 ? 1 : 0
          );
          IFDBG(filtered_num_actual_neg_gain_moves += actual_gain < 0 ? 1 : 0);
          IFDBG(filtered_actual_gain_sum += actual_gain);

          p_graph.set_block(u, to);
          gain_cache.move(p_graph, u, from, p_graph.block(u));

          if (p_graph.degree(u) >= 0.001 * p_graph.n()) {
            IFDBG(++filtered_num_moves_deg01perc);
          }
          if (p_graph.degree(u) >= 0.01 * p_graph.n()) {
            IFDBG(++filtered_num_moves_deg1perc);
          }
          if (p_graph.degree(u) >= 50'000) {
            IFDBG(++filtered_num_moves_deg50k);
          }
        }
      });
    };

    const EdgeWeight pre_rebalance_cut = IFDBG(metrics::edge_cut(p_graph));
    const double pre_rebalance_balance = IFDBG(metrics::imbalance(p_graph));
    const bool pre_rebalance_feasible =
        IFDBG(metrics::is_feasible(p_graph, p_ctx));

    StaticArray<BlockID> imbalanced_partition(IFDBG(p_graph.n()));
    if constexpr (kDebug) {
      p_graph.pfor_nodes([&](const NodeID u) {
        imbalanced_partition[u] = p_graph.block(u);
      });
    }

    TIMED_SCOPE("Rebalance") {
      balancer.refine(p_graph, p_ctx);
    };

    if constexpr (kDebug) {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (imbalanced_partition[u] != p_graph.block(u)) {
          IFDBG(++rebalance_num_moves);
          IFDBG(final_num_doublemoved += lock[u]);
          if (p_graph.degree(u) >= 0.001 * p_graph.n()) {
            IFDBG(++rebalance_num_moves_deg01perc);
            IFDBG(final_num_doublemoved_deg01perc += lock[u]);
          }
          if (p_graph.degree(u) >= 0.01 * p_graph.n()) {
            IFDBG(++rebalance_num_moves_deg1perc);
            IFDBG(final_num_doublemoved_deg1perc += lock[u]);
          }
          if (p_graph.degree(u) >= 50'000) {
            IFDBG(++rebalance_num_moves_deg50k);
            IFDBG(final_num_doublemoved_deg50k += lock[u]);
          }
        }
      });
    }

    TIMED_SCOPE("Update best partition") {
      const EdgeWeight current_cut = metrics::edge_cut(p_graph);
      if (current_cut <= best_cut) {
        p_graph.pfor_nodes([&](const NodeID u) {
          best_partition[u] = p_graph.block(u);
        });
        best_cut = current_cut;
        last_iteration_is_best = true;
      } else {
        last_iteration_is_best = false;
      }
    };

    const EdgeWeight post_rebalance_cut = IFDBG(metrics::edge_cut(p_graph));
    const double post_rebalance_balance = IFDBG(metrics::imbalance(p_graph));
    const bool post_rebalance_feasible =
        IFDBG(metrics::is_feasible(p_graph, p_ctx));

    DBG << "iter=" << i << " "
        << "pre_rebalance_cut=" << pre_rebalance_cut << " "
        << "pre_rebalance_balance=" << pre_rebalance_balance << " "
        << "pre_rebalance_feasible=" << pre_rebalance_feasible << " "
        << "post_rebalance_cut=" << post_rebalance_cut << " "
        << "post_rebalance_balance=" << post_rebalance_balance << " "
        << "post_rebalance_feasible=" << post_rebalance_feasible << " "
        << "initial_pos_gain_sum=" << initial_pos_gain_sum << " "
        << "initial_gain_sum=" << initial_gain_sum << " "
        << "initial_num_pos_gain_moves=" << initial_num_pos_gain_moves << " "
        << "initial_num_moves=" << initial_num_moves << " "
        << "prefiltered_gain_sum=" << prefiltered_gain_sum << " "
        << "prefiltered_num_moves=" << prefiltered_num_moves << " "
        << "filtered_expected_gain_sum=" << filtered_expected_gain_sum << " "
        << "filtered_actual_gain_sum=" << filtered_actual_gain_sum << " "
        << "filtered_num_actual_pos_gain_moves="
        << filtered_num_actual_pos_gain_moves << " "
        << "filtered_num_actual_zero_gain_moves="
        << filtered_num_actual_zero_gain_moves << " "
        << "filtered_num_actual_neg_gain_moves="
        << filtered_num_actual_neg_gain_moves << " "
        << "filtered_num_moves=" << filtered_num_moves << " "
        << "filtered_num_moves_deg01perc=" << filtered_num_moves_deg01perc
        << " "
        << "filtered_num_moves_deg1perc=" << filtered_num_moves_deg1perc << " "
        << "filtered_num_moves_deg50k=" << filtered_num_moves_deg50k << " "
        << "rebalance_num_moves=" << rebalance_num_moves << " "
        << "rebalance_num_moves_deg01perc=" << rebalance_num_moves_deg01perc
        << " "
        << "rebalance_num_moves_deg1perc=" << rebalance_num_moves_deg1perc
        << " "
        << "rebalance_num_moves_deg50k=" << rebalance_num_moves_deg50k << " "
        << "final_num_doublemoved=" << final_num_doublemoved << " "
        << "final_num_doublemoved_deg01perc=" << final_num_doublemoved_deg01perc
        << " "
        << "final_num_doublemoved_deg1perc=" << final_num_doublemoved_deg1perc
        << " "
        << "final_num_doublemoved_deg50k=" << final_num_doublemoved_deg50k
        << " "
        << "last_iteration_is_best=" << last_iteration_is_best;
  }

  TIMED_SCOPE("Rollback") {
    if (!last_iteration_is_best) {
      p_graph.pfor_nodes([&](const NodeID u) {
        p_graph.set_block(u, best_partition[u]);
      });
    }
  };

  return false;
}
} // namespace kaminpar::shm
