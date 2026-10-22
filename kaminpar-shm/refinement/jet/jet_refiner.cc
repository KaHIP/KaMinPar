/*******************************************************************************
 * Shared-memory implementation of JET, due to
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 *
 * @file:   jet_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/jet/jet_refiner.h"

#include <memory>
#include <type_traits>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"
#include "kaminpar-shm/refinement/gains/jet_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"
#include "kaminpar-shm/refinement/jet/jet_balancer.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename Graph> class JetRefinerImpl {
  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

public:
  JetRefinerImpl(const Context &ctx) : _ctx(ctx) {}

  JetRefinerImpl(const JetRefinerImpl &) = delete;
  JetRefinerImpl &operator=(const JetRefinerImpl &) = delete;

  JetRefinerImpl(JetRefinerImpl &&) noexcept = default;
  JetRefinerImpl &operator=(JetRefinerImpl &&) = default;

  void initialize(const PartitionedGraph &p_graph) {
    SCOPED_TIMER("Jet Refiner");
    SCOPED_TIMER("Initialization");

    const bool is_coarse_level = p_graph.graph().level() > 0;

    if (is_coarse_level) {
      _num_rounds = _ctx.refinement.jet.num_rounds_on_coarse_level;
      _initial_gain_temp = _ctx.refinement.jet.initial_gain_temp_on_coarse_level;
      _final_gain_temp = _ctx.refinement.jet.final_gain_temp_on_coarse_level;
    } else {
      _num_rounds = _ctx.refinement.jet.num_rounds_on_fine_level;
      _initial_gain_temp = _ctx.refinement.jet.initial_gain_temp_on_fine_level;
      _final_gain_temp = _ctx.refinement.jet.final_gain_temp_on_fine_level;
    }

    DBG << "Initialized Jet refiner: " << _num_rounds << " rounds, gain temperature in range ["
        << _initial_gain_temp << ", " << _final_gain_temp << "]";
  }

  void refine(PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx) {
    switch (_ctx.refinement.jet.gain_cache_strategy) {
    case JetGainCacheStrategy::SPARSE:
      refine_with_cache<NormalSparseGainCache<Graph>>(p_graph, graph, p_ctx);
      break;
    case JetGainCacheStrategy::JET:
      refine_with_cache<SynchronousJetGainCache<Graph>>(p_graph, graph, p_ctx);
      break;
    }
  }

private:
  template <typename GainCache>
  void
  refine_with_cache(PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("Jet Refiner");

    START_TIMER("Allocation");
    GainCache gain_cache(_ctx, p_graph.graph().n(), p_graph.k());
    gain_cache.initialize(graph, p_graph);

    StaticArray<BlockID> next_partition(p_graph.graph().n());
    graph.pfor_nodes([&](const NodeID u) { next_partition[u] = 0; });

    StaticArray<std::uint8_t> lock(p_graph.graph().n());
    graph.pfor_nodes([&](const NodeID u) { lock[u] = 0; });

    BlockParallelOverloadBalancer overload_balancer(_ctx);
    JetBalancer<Graph, GainCache> jet_balancer(_ctx);
    const bool use_jet_balancer =
        _ctx.refinement.jet.balancing_algorithm == JetBalancingAlgorithm::JET;
    auto initialize_balancer = [&] {
      if (use_jet_balancer) {
        jet_balancer.initialize(p_graph);
      } else {
        overload_balancer.initialize(p_graph);
        overload_balancer.track_moves([&](const NodeID u, const BlockID from, const BlockID to) {
          gain_cache.move(u, from, to);
        });
      }
    };
    initialize_balancer();

    StaticArray<BlockID> best_partition(p_graph.graph().n());
    graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });

    const auto balance_violation = [&] {
      BlockWeight violation = metrics::total_overload(p_graph, p_ctx);
      if (p_ctx.has_min_block_weights()) {
        for (const BlockID b : p_graph.blocks()) {
          violation +=
              std::max<BlockWeight>(0, p_ctx.min_block_weight(b) - p_graph.block_weight(b));
        }
      }
      return violation;
    };
    const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
    EdgeWeight best_cut = initial_cut;
    BlockWeight best_violation = balance_violation();
    bool best_is_feasible = best_violation == 0;
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

    for (int round = 0; round < _num_rounds; ++round) {
      int cur_fruitless_iteration = 0;
      int cur_iteration = 0;

      const double gain_temp = compute_gain_temp(round);
      DBG << "Round " << (round + 1) << " of " << _num_rounds << ": use gain temperature "
          << gain_temp;

      do {
        TIMED_SCOPE("Find moves") {
          graph.pfor_nodes([&](const NodeID u) {
            const BlockID from = p_graph.block(u);

            if (lock[u] || !gain_cache.is_border_node(u, from)) {
              next_partition[u] = from;
              return;
            }

            EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
            BlockID best_block = from;

            if constexpr (std::is_same_v<GainCache, SynchronousJetGainCache<Graph>>) {
              const EdgeWeight conn_from = gain_cache.conn(u, from);
              gain_cache.gains(u, from, [&](const BlockID to, auto &&conn) {
                const EdgeWeight gain = conn() - conn_from;
                // Preserve the legacy smallest-block tie break independently of hash order.
                if (gain > best_gain || (gain == best_gain && to < best_block)) {
                  best_gain = gain;
                  best_block = to;
                }
              });
            } else {
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
            }

            if (best_gain > -std::floor(gain_temp * gain_cache.conn(u, from))) {
              next_partition[u] = best_block;
            } else {
              next_partition[u] = from;
            }
          });
        };

        TIMED_SCOPE("Filter moves") {
          graph.pfor_nodes([&](const NodeID u) {
            lock[u] = 0;

            const BlockID from = p_graph.block(u);
            const BlockID to = next_partition[u];
            if (from == to) {
              return;
            }

            const EdgeWeight gain_u = gain_cache.gain(u, from, to);
            EdgeWeight gain = 0;

            graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
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
            });

            if (gain > 0) {
              lock[u] = 1;
            }
          });
        };

        TIMED_SCOPE("Execute moves") {
          const auto execute_move = [&](const NodeID u) {
            if (lock[u]) {
              const BlockID from = p_graph.block(u);
              const BlockID to = next_partition[u];

              if (p_ctx.has_min_block_weights() &&
                  p_graph.block_weight(from) - graph.node_weight(u) <
                      p_ctx.min_block_weight(from)) {
                lock[u] = 0;
                return;
              }
              p_graph.set_block(u, to);
              gain_cache.move(u, from, to);
            }
          };
          if (p_ctx.has_min_block_weights()) {
            // The existing block-weight API does not atomically reserve source capacity.
            // Serialize this optional path so simultaneous departures cannot violate a minimum.
            for (const NodeID u : graph.nodes()) {
              execute_move(u);
            }
          } else {
            graph.pfor_nodes(execute_move);
          }
        };

        TIMED_SCOPE("Rebalance") {
          if (use_jet_balancer) {
            jet_balancer.refine(p_graph, graph, p_ctx, gain_cache);
          } else {
            overload_balancer.refine(p_graph, p_ctx);
          }
        };

        const EdgeWeight final_cut = metrics::edge_cut(p_graph);
        const BlockWeight violation = balance_violation();
        const bool feasible = violation == 0;
        const bool improved_balance = !best_is_feasible && violation < best_violation;
        const bool accept =
            improved_balance || (feasible && (!best_is_feasible || final_cut <= best_cut));

        TIMED_SCOPE("Update best partition") {
          if (accept) {
            graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
            last_iteration_is_best = true;
          } else {
            last_iteration_is_best = false;
          }
        };

        ++cur_iteration;
        ++cur_fruitless_iteration;

        if (improved_balance ||
            (feasible &&
             best_cut - final_cut > (1.0 - _ctx.refinement.jet.fruitless_threshold) * best_cut)) {
          DBG << "Improved cut from " << initial_cut << " to " << best_cut << " to " << final_cut
              << ": resetting number of fruitless iterations (threshold: "
              << _ctx.refinement.jet.fruitless_threshold << ")";
          cur_fruitless_iteration = 0;
        } else {
          DBG << "Fruitless edge cut change from " << initial_cut << " to " << best_cut << " to "
              << final_cut << " (threshold: " << _ctx.refinement.jet.fruitless_threshold
              << "): incrementing fruitless iterations counter to " << cur_fruitless_iteration;
        }

        if (accept) {
          best_cut = final_cut;
          best_violation = violation;
          best_is_feasible = feasible;
        }
      } while (cur_iteration < max_num_iterations &&
               cur_fruitless_iteration < max_num_fruitless_iterations);

      TIMED_SCOPE("Rollback") {
        if (!last_iteration_is_best) {
          graph.pfor_nodes([&](const NodeID u) { p_graph.set_block(u, best_partition[u]); });

          // Re-initialize gain cache and balancer after rollback
          gain_cache.initialize(graph, p_graph);
          initialize_balancer();
        }
      };
    }
  }

private:
  [[nodiscard]] double compute_gain_temp(int round) const {
    if (_num_rounds > 1) {
      const double alpha = 1.0 * round / (_num_rounds - 1);
      return _initial_gain_temp + alpha * (_final_gain_temp - _initial_gain_temp);
    }

    return 1.0 * (_initial_gain_temp + _final_gain_temp) / 2;
  }

private:
  const Context &_ctx;

  int _num_rounds = 0;
  double _initial_gain_temp = 0.0;
  double _final_gain_temp = 0.0;
};

JetRefiner::JetRefiner(const Context &ctx)
    : _ctx(ctx),
      _impl(std::make_unique<ReifiedGraphComponent<JetRefinerImpl>>()) {}

JetRefiner::~JetRefiner() = default;

std::string JetRefiner::name() const {
  return "Jet";
}

void JetRefiner::initialize(const PartitionedGraph &p_graph) {
  reified(p_graph, [&]<typename Graph>(const Graph &) {
    _impl->ensure<Graph>(_ctx).initialize(p_graph);
  });
}

bool JetRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  _impl->with(p_graph, [&](auto &impl, const auto &graph) {
    return impl.refine(p_graph, graph, p_ctx);
  });

  return false; // (ignored)
}

} // namespace kaminpar::shm
