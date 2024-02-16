/*******************************************************************************
 * Shared-memory implementation of JET, due to
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 *
 * @file:   jet_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/jet/jet_refiner.h"

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/greedy_balancer.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

void JetRefiner::initialize(const PartitionedGraph &p_graph) {
  SCOPED_TIMER("Jet Refiner");
  SCOPED_TIMER("Initialization");

  _negative_gain_factor = (p_graph.n() == _ctx.partition.n)
                              ? _ctx.refinement.jet.fine_negative_gain_factor
                              : _ctx.refinement.jet.coarse_negative_gain_factor;
  DBG << "Initialized with negative gain factor: " << _negative_gain_factor;
}

bool JetRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Jet Refiner");

  START_TIMER("Allocation");
  SparseGainCache<true> gain_cache(_ctx, p_graph.n(), p_graph.k());
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
  const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
  EdgeWeight best_cut = initial_cut;
  bool last_iteration_is_best = true;
  STOP_TIMER();

  const int max_num_fruitless_iterations = (_ctx.refinement.jet.num_fruitless_iterations == 0)
                                               ? std::numeric_limits<int>::max()
                                               : _ctx.refinement.jet.num_fruitless_iterations;
  const int max_num_iterations = (_ctx.refinement.jet.num_iterations == 0)
                                     ? std::numeric_limits<int>::max()
                                     : _ctx.refinement.jet.num_iterations;
  DBG << "Running JET refinement for at most " << max_num_iterations << " iterations and at most "
      << max_num_fruitless_iterations << " fruitless iterations";

  int cur_fruitless_iteration = 0;
  int cur_iteration = 0;

  do {
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

        if (best_gain > -std::floor(_negative_gain_factor * gain_cache.conn(u, from))) {
          next_partition[u] = best_block;
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
          lock[u] = 1;
        }
      });
    };

    TIMED_SCOPE("Execute moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (lock[u]) {
          const BlockID from = p_graph.block(u);
          const BlockID to = next_partition[u];

          p_graph.set_block(u, to);
          gain_cache.move(p_graph, u, from, to);
        }
      });
    };

    TIMED_SCOPE("Rebalance") {
      balancer.refine(p_graph, p_ctx);
    };

    const EdgeWeight final_cut = metrics::edge_cut(p_graph);

    TIMED_SCOPE("Update best partition") {
      if (final_cut <= best_cut) {
        p_graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
        last_iteration_is_best = true;
      } else {
        last_iteration_is_best = false;
      }
    };

    ++cur_iteration;
    ++cur_fruitless_iteration;

    if (best_cut - final_cut > (1.0 - _ctx.refinement.jet.fruitless_threshold) * best_cut) {
      DBG << "Improved cut from " << initial_cut << " to " << best_cut << " to " << final_cut
          << ": resetting number of fruitless iterations (threshold: "
          << _ctx.refinement.jet.fruitless_threshold << ")";
      cur_fruitless_iteration = 0;
    } else {
      DBG << "Fruitless edge cut change from " << initial_cut << " to " << best_cut << " to "
          << final_cut << " (threshold: " << _ctx.refinement.jet.fruitless_threshold
          << "): incrementing fruitless iterations counter to " << cur_fruitless_iteration;
    }

    if (final_cut < best_cut) {
      best_cut = final_cut;
    }
  } while (cur_iteration < max_num_iterations &&
           cur_fruitless_iteration < max_num_fruitless_iterations);

  TIMED_SCOPE("Rollback") {
    if (!last_iteration_is_best) {
      p_graph.pfor_nodes([&](const NodeID u) { p_graph.set_block(u, best_partition[u]); });
    }
  };

  return false;
}
} // namespace kaminpar::shm
