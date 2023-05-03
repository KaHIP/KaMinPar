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
    parallel::Atomic<EdgeWeight> sum_found_pos_moves = 0;
    parallel::Atomic<EdgeWeight> sum_found_moves = 0;
    parallel::Atomic<EdgeWeight> num_found_pos_moves = 0;
    parallel::Atomic<EdgeWeight> sum_actual_moves = 0;
    parallel::Atomic<EdgeWeight> num_actual_pos_moves = 0;

    parallel::Atomic<NodeID> num_premoved = 0;
    parallel::Atomic<NodeID> num_premoved_01 = 0;
    parallel::Atomic<NodeID> num_premoved_1 = 0;
    parallel::Atomic<NodeID> num_premoved_50k = 0;

    parallel::Atomic<NodeID> num_backmoved = 0;
    parallel::Atomic<NodeID> num_backmoved_01 = 0;
    parallel::Atomic<NodeID> num_backmoved_1 = 0;
    parallel::Atomic<NodeID> num_backmoved_50k = 0;

    parallel::Atomic<NodeID> num_backmoved_org = 0;

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

        if (-best_gain < std::floor(c * gain_cache.conn(u, from))) {
          next_partition[u] = best_block;

          IFDBG(sum_found_moves += best_gain);
          if (best_gain >= 0) {
            IFDBG(sum_found_pos_moves += best_gain);
            IFDBG(++num_found_pos_moves);
          }
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
          lock[u] = 1;
        }
      });
    };

    TIMED_SCOPE("Execute moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (lock[u]) {
          const BlockID from = p_graph.block(u);
          const BlockID to = next_partition[u];

          if (IFDBG(gain_cache.gain(u, from, to) > 0)) {
            IFDBG(++num_actual_pos_moves);
            IFDBG(sum_actual_moves += gain_cache.gain(u, from, to));
          }

          p_graph.set_block(u, to);
          gain_cache.move(p_graph, u, from, p_graph.block(u));

          IFDBG(++num_premoved);
          if (p_graph.degree(u) >= 0.001 * p_graph.n()) {
            IFDBG(++num_premoved_01);
          }
          if (p_graph.degree(u) >= 0.01 * p_graph.n()) {
            IFDBG(++num_premoved_1);
          }
          if (p_graph.degree(u) >= 50'000) {
            IFDBG(++num_premoved_50k);
          }
        }
      });
    };

    TIMED_SCOPE("Statistics") {
      const EdgeWeight pre_rebalance_cut = IFDBG(metrics::edge_cut(p_graph));
      const double pre_rebalance_balance = IFDBG(metrics::imbalance(p_graph));
      const bool pre_rebalance_feasible =
          IFDBG(metrics::is_feasible(p_graph, p_ctx));
      DBG << "After iteration " << i
          << ", pre-rebalance: cut=" << pre_rebalance_cut
          << ", imbalance=" << pre_rebalance_balance
          << ", feasible=" << pre_rebalance_feasible
          << ", sum_found_pos_moves=" << sum_found_pos_moves
          << ", sum_found_moves=" << sum_found_moves
          << ", num_found_pos_moves=" << num_found_pos_moves
          << ", sum_actual_moves=" << sum_actual_moves
          << ", num_actual_pos_moves=" << num_actual_pos_moves
          << ", num_premoved=" << num_premoved
          << ", num_premoved_01=" << num_premoved_01
          << ", num_premoved_1=" << num_premoved_1
          << ", num_premoved_50k=" << num_premoved_50k;
    };

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
          IFDBG(++num_backmoved);
          if (p_graph.degree(u) >= 0.001 * p_graph.n()) {
            IFDBG(++num_backmoved_01);
          }
          if (p_graph.degree(u) >= 0.01 * p_graph.n()) {
            IFDBG(++num_backmoved_1);
          }
          if (p_graph.degree(u) >= 50'000) {
            IFDBG(++num_backmoved_50k);
          }
          if (lock[u]) {
            IFDBG(++num_backmoved_org);
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

    TIMED_SCOPE("Statistics") {
      const EdgeWeight post_rebalance_cut = IFDBG(metrics::edge_cut(p_graph));
      const double post_rebalance_balance = IFDBG(metrics::imbalance(p_graph));
      const bool post_rebalance_feasible =
          IFDBG(metrics::is_feasible(p_graph, p_ctx));
      DBG << "After iteration " << i
          << ", post-rebalance: cut=" << post_rebalance_cut
          << ", imbalance=" << post_rebalance_balance
          << ", feasible=" << post_rebalance_feasible
          << ", num_backmoved=" << num_backmoved
          << ", num_backmoved_01=" << num_backmoved_01
          << ", num_backmoved_1=" << num_backmoved_1
          << ", num_backmoved_50k=" << num_backmoved_50k
          << ", num_backmoved_org=" << num_backmoved_org;
    };
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
