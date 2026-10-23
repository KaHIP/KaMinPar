/*******************************************************************************
 * Shared-memory implementation of JET, due to
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 *
 * @file:   jet_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/jet/jet_refiner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"
#include "kaminpar-shm/refinement/jet/jet_balancer.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

// The paper balancer updates its cache directly. Also record those moves for
// incremental cut accounting, including vertices moved repeatedly in one pass.
template <typename GainCache, typename MoveTracker> class TrackedGainCache {
public:
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  TrackedGainCache(GainCache &cache, const MoveTracker &track_move)
      : _cache(cache),
        _track_move(track_move) {}

  EdgeWeight conn(const NodeID u, const BlockID block) const {
    return _cache.conn(u, block);
  }

  template <typename Lambda> void gains(const NodeID u, const BlockID from, Lambda &&lambda) const {
    _cache.gains(u, from, std::forward<Lambda>(lambda));
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    _cache.move(u, from, to);
    _track_move(u, from, to);
  }

private:
  GainCache &_cache;
  const MoveTracker &_track_move;
};

template <typename Graph> class JetRefinerImpl {
  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

  using SparseCache = SparseGainCache<Graph, GenericDeltaGainCache, false>;
  using OnTheFlyCache = OnTheFlyGainCache<Graph, false, false, true>;

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

  bool refine(PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("Jet Refiner");
    if (_num_rounds <= 0 || p_graph.k() < 2) {
      return false;
    }
    const auto strategy = _ctx.refinement.jet.gain_cache_strategy;
    if (strategy != GainCacheStrategy::SPARSE && strategy != GainCacheStrategy::ON_THE_FLY) {
      throw std::invalid_argument("Jet supports only sparse and on-the-fly gain caches");
    }
    // Sparse rows require degree ordering and weights that fit the packed entries.
    if (strategy == GainCacheStrategy::ON_THE_FLY || !can_use_sparse_cache(graph, p_graph.k())) {
      return refine_with_cache<OnTheFlyCache>(p_graph, graph, p_ctx);
    }
    return refine_with_cache<SparseCache>(p_graph, graph, p_ctx);
  }

private:
  bool can_use_sparse_cache(const Graph &graph, const BlockID k) const {
    if (!graph.sorted()) {
      return false;
    }
    const UnsignedEdgeWeight packed_limit =
        std::numeric_limits<UnsignedEdgeWeight>::max() >> math::ceil_log2(k);
    const EdgeID degree_threshold = std::max<EdgeID>(
        k * _ctx.refinement.kway_fm.k_based_high_degree_threshold,
        _ctx.refinement.kway_fm.constant_high_degree_threshold
    );
    // SparseGainCache assigns complete degree buckets to packed hash rows.
    NodeID packed_nodes = 0;
    for (std::size_t bucket = 0;
         packed_nodes < graph.n() && graph.degree(packed_nodes) < degree_threshold;
         ++bucket) {
      packed_nodes += graph.bucket_size(bucket);
    }
    return tbb::parallel_reduce(
        tbb::blocked_range<NodeID>(0, packed_nodes),
        true,
        [&](const auto &range, bool safe) {
          for (NodeID u = range.begin(); u != range.end(); ++u) {
            if (!safe) {
              break;
            }
            UnsignedEdgeWeight weighted_degree = 0;
            graph.adjacent_nodes(u, [&](NodeID, const EdgeWeight weight) {
              weighted_degree += static_cast<UnsignedEdgeWeight>(weight);
            });
            safe = weighted_degree <= packed_limit;
          }
          return safe;
        },
        std::logical_and<>{}
    );
  }

  template <typename GainCache>
  bool
  refine_with_cache(PartitionedGraph &p_graph, const Graph &graph, const PartitionContext &p_ctx) {
    START_TIMER("Allocation");
    auto &stored_cache = std::get<std::unique_ptr<GainCache>>(_gain_caches);
    if (!stored_cache) {
      stored_cache = std::make_unique<GainCache>(_ctx, graph.n(), p_graph.k());
    }
    GainCache &gain_cache = *stored_cache;
    gain_cache.initialize(graph, p_graph);
    jet::DestinationCache destinations(gain_cache, graph.n());

    StaticArray<BlockID> next_partition(graph.n(), static_array::noinit);
    StaticArray<EdgeWeight> candidate_gain(graph.n(), static_array::noinit);
    StaticArray<std::uint8_t> lock(graph.n(), std::uint8_t{0});

    // A vertex can move in both LP and rebalancing. Remember its original part
    // only once, then account for all changed edges after both phases finish.
    StaticArray<BlockID> old_block(graph.n(), kInvalidBlockID);
    tbb::enumerable_thread_specific<std::vector<NodeID>> moved_nodes;
    const auto track_move = [&](const NodeID u, const BlockID from, BlockID) {
      destinations.invalidate_neighborhood(graph, u);
      // Both phases have a single owner per vertex; their barriers separate them.
      if (old_block[u] == kInvalidBlockID) {
        old_block[u] = from;
        moved_nodes.local().push_back(u);
      }
    };

    TrackedGainCache tracked_cache(gain_cache, track_move);
    BlockParallelOverloadBalancer overload_balancer(_ctx);
    JetBalancer<Graph, decltype(tracked_cache)> jet_balancer(_ctx);
    const bool use_jet_balancer =
        _ctx.refinement.jet.balancing_algorithm == JetBalancingAlgorithm::JET;
    const auto initialize_balancer = [&] {
      if (use_jet_balancer) {
        jet_balancer.initialize(p_graph);
      } else {
        overload_balancer.initialize(p_graph);
        overload_balancer.track_moves([&](const NodeID u, const BlockID from, const BlockID to) {
          tracked_cache.move(u, from, to);
        });
      }
    };
    initialize_balancer();

    StaticArray<BlockID> best_partition(graph.n(), static_array::noinit);
    graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });

    const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
    EdgeWeight best_cut = initial_cut;
    std::int64_t current_cut = initial_cut;
    const auto total_violation = [&] {
      // Overload and underload can each approach the total node weight, so their
      // sum needs the unsigned range even when individual block weights fit.
      UnsignedNodeWeight violation = 0;
      for (const BlockID b : p_graph.blocks()) {
        const BlockWeight weight = p_graph.block_weight(b);
        violation += std::max<BlockWeight>(0, weight - p_ctx.max_block_weight(b));
        violation += std::max<BlockWeight>(0, p_ctx.min_block_weight(b) - weight);
      }
      return violation;
    };
    const UnsignedNodeWeight initial_violation = total_violation();
    UnsignedNodeWeight best_violation = initial_violation;
    bool last_iteration_is_best = true;
    bool previous_lp_moved = false;
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
        const bool had_locked_nodes = previous_lp_moved;
        TIMED_SCOPE("Find moves") {
          graph.pfor_nodes([&](const NodeID u) {
            const BlockID from = p_graph.block(u);

            next_partition[u] = from;
            candidate_gain[u] = std::numeric_limits<EdgeWeight>::min();
            if (lock[u] || (p_ctx.has_min_block_weights() &&
                            p_graph.block_weight(from) - graph.node_weight(u) <
                                p_ctx.min_block_weight(from))) {
              return;
            }

            const auto &destination = destinations.get(u, from);
            if (destination.accepts(gain_temp)) {
              next_partition[u] = destination.target;
              candidate_gain[u] = destination.gain();
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

            const EdgeWeight gain_u = candidate_gain[u];
            EdgeWeight gain = 0;

            graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
              const EdgeWeight gain_v = candidate_gain[v];
              const bool v_before_u = gain_v > gain_u || (gain_v == gain_u && v < u);
              const BlockID block_v = v_before_u ? next_partition[v] : p_graph.block(v);

              if (to == block_v) {
                gain += weight;
              } else if (from == block_v) {
                gain -= weight;
              }
            });

            if (gain >= 0) {
              lock[u] = 1;
            }
          });
        };

        TIMED_SCOPE("Execute moves") {
          const auto execute_move = [&](const NodeID u) {
            if (lock[u]) {
              const BlockID from = p_graph.block(u);
              const BlockID to = next_partition[u];
              if (!p_ctx.has_min_block_weights()) {
                p_graph.set_block(u, to);
              } else if (!p_graph.move(
                             u,
                             from,
                             to,
                             std::numeric_limits<BlockWeight>::max(),
                             p_ctx.min_block_weight(from)
                         )) {
                lock[u] = 0;
                return;
              }
              tracked_cache.move(u, from, to);
            }
          };
          graph.pfor_nodes(execute_move);
        };

        previous_lp_moved =
            std::any_of(moved_nodes.begin(), moved_nodes.end(), [](const auto &nodes) {
              return !nodes.empty();
            });

        TIMED_SCOPE("Rebalance") {
          if (use_jet_balancer) {
            jet_balancer.refine(p_graph, graph, p_ctx, tracked_cache);
          } else {
            overload_balancer.refine(p_graph, p_ctx);
          }
        };

        const bool any_moves =
            std::any_of(moved_nodes.begin(), moved_nodes.end(), [](const auto &nodes) {
              return !nodes.empty();
            });
        TIMED_SCOPE("Update cut") {
          const std::int64_t gain = tbb::parallel_reduce(
              moved_nodes.range(),
              std::int64_t{0},
              [&](const auto &range, std::int64_t sum) {
                for (const auto &nodes : range) {
                  for (const NodeID u : nodes) {
                    const BlockID from = old_block[u];
                    const BlockID to = p_graph.block(u);
                    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
                      const BlockID old_v = old_block[v];
                      // Count edges between two moved vertices only once.
                      if (old_v == kInvalidBlockID || u < v) {
                        const BlockID final_v = p_graph.block(v);
                        const BlockID original_v = old_v == kInvalidBlockID ? final_v : old_v;
                        sum += static_cast<std::int64_t>(weight) *
                               (static_cast<int>(from != original_v) -
                                static_cast<int>(to != final_v));
                      }
                    });
                  }
                }
                return sum;
              },
              std::plus<>{}
          );
          current_cut -= gain;
          KASSERT(current_cut >= 0 && current_cut <= std::numeric_limits<EdgeWeight>::max());
          KASSERT(
              current_cut == metrics::edge_cut(p_graph), "incorrect incremental cut", assert::heavy
          );

          // Do not clear the markers until all readers in the reduction finish.
          tbb::parallel_for(moved_nodes.range(), [&](const auto &range) {
            for (auto &nodes : range) {
              for (const NodeID u : nodes) {
                old_block[u] = kInvalidBlockID;
              }
              nodes.clear();
            }
          });
        };
        const EdgeWeight final_cut = static_cast<EdgeWeight>(current_cut);
        const UnsignedNodeWeight violation = total_violation();
        const bool improved_balance = violation < best_violation;
        const bool improved_cut = violation == best_violation && final_cut < best_cut;
        const bool significant_improvement =
            improved_balance ||
            (improved_cut &&
             best_cut - final_cut > (1.0 - _ctx.refinement.jet.fruitless_threshold) * best_cut);

        TIMED_SCOPE("Update best partition") {
          if (improved_balance || improved_cut) {
            graph.pfor_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
            best_cut = final_cut;
            best_violation = violation;
            last_iteration_is_best = true;
          } else if (any_moves) {
            last_iteration_is_best = false;
          }
        };

        ++cur_iteration;
        ++cur_fruitless_iteration;

        if (significant_improvement) {
          DBG << "Improved cut from " << initial_cut << " to " << best_cut << " to " << final_cut
              << ": resetting number of fruitless iterations (threshold: "
              << _ctx.refinement.jet.fruitless_threshold << ")";
          cur_fruitless_iteration = 0;
        } else {
          DBG << "Fruitless edge cut change from " << initial_cut << " to " << best_cut << " to "
              << final_cut << " (threshold: " << _ctx.refinement.jet.fruitless_threshold
              << "): incrementing fruitless iterations counter to " << cur_fruitless_iteration;
        }

        // A no-move pass may merely have cleared the preceding LP locks. Allow
        // one more pass in that case before declaring the round converged.
        if (!any_moves && !had_locked_nodes) {
          break;
        }
      } while (cur_iteration < max_num_iterations &&
               cur_fruitless_iteration < max_num_fruitless_iterations);

      TIMED_SCOPE("Rollback") {
        if (!last_iteration_is_best) {
          graph.pfor_nodes([&](const NodeID u) {
            if (p_graph.block(u) != best_partition[u]) {
              p_graph.set_block(u, best_partition[u]);
            }
            lock[u] = 0;
          });
          gain_cache.initialize(graph, p_graph);
          destinations.invalidate_all(graph);
          initialize_balancer();
          current_cut = best_cut;
          last_iteration_is_best = true;
          previous_lp_moved = false;
        }
      };
    }

    return best_violation < initial_violation ||
           (best_violation == initial_violation && best_cut < initial_cut);
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
  // Retain allocations across refinement calls and uncoarsening levels. Each
  // backend refreshes its graph/partition pointers when it is initialized.
  std::tuple<std::unique_ptr<SparseCache>, std::unique_ptr<OnTheFlyCache>> _gain_caches;

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
  return _impl->with(p_graph, [&](auto &impl, const auto &graph) {
    return impl.refine(p_graph, graph, p_ctx);
  });
}

} // namespace kaminpar::shm
