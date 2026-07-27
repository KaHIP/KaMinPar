/*******************************************************************************
 * Parallel unconstrained k-way label propagation refiner.
 *
 * @file:   unconstrained_lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   20.05.2026
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/unconstrained_lp_refiner.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/positive_gain_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/multi_queue_overload_balancer.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename Graph> class UnconstrainedLPRefinerImpl {
  using Gain = std::int64_t;
  using RatingMaps = lp::RatingMapPool<BlockID, Gain, lp::adaptive_rating_map::SparseMap>;
  using RatingMap = typename RatingMaps::RatingMap;

  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  explicit UnconstrainedLPRefinerImpl(const Context &ctx)
      : _lp_ctx(ctx.refinement.lp),
        _balancer(ctx) {}

  void initialize(const Graph *graph) {
    _graph = graph;
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    KASSERT(_graph == p_graph.graph().underlying_graph());
    KASSERT(p_graph.k() <= p_ctx.k);
    SCOPED_HEAP_PROFILER("Unconstrained Label Propagation");

    allocate(p_graph);

    _balancer.initialize(p_graph);
    _balancer.track_moves([&](const NodeID u, const BlockID from, const BlockID) {
      record_moved_node(u, from);
    });

    Gain cut_before = compute_edge_cut(p_graph);
    NodeID active_nodes = initialize_active_nodes(p_graph);
    bool found_improvement = false;

    const std::size_t max_iterations =
        _lp_ctx.num_iterations == 0 ? kInfiniteIterations : _lp_ctx.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));

      if (active_nodes == 0) {
        break;
      }

      clear_moved_nodes();

      const NodeID num_moves = perform_round(p_graph);
      if (num_moves == 0) {
        break;
      }

      if (metrics::total_overload(p_graph, p_ctx) > 0) {
        TIMED_SCOPE("Rebalance") {
          _balancer.refine(p_graph, p_ctx);
        };
      }

      const Gain improvement = TIMED_SCOPE("Compute improvement") {
        return compute_round_improvement(p_graph);
      };
      if (metrics::total_overload(p_graph, p_ctx) > 0 || improvement < 0) {
        restore_partition(p_graph);
        break;
      }

      activate_moved_nodes();

      const Gain previous_cut = cut_before;
      cut_before = std::max<Gain>(0, cut_before - improvement);
      const double relative_improvement =
          previous_cut == 0 ? 0.0 : 1.0 * improvement / previous_cut;

      found_improvement = true;
      active_nodes = update_active_nodes();

      if (relative_improvement < _lp_ctx.unconstrained_min_improvement_factor) {
        break;
      }
    }

    return found_improvement;
  }

  void set_communities(std::span<const NodeID> communities) {
    _communities = communities;
  }

private:
  void allocate(const PartitionedGraph &p_graph) {
    const NodeID n = p_graph.n();
    if (_active.size() != n) {
      _active.resize(n);
      _next_active.resize(n);
      _moved.resize(n);
      _round_start_partition.resize(n, ::kaminpar::static_array::noinit);
    }

    _rating_maps.ensure_capacity(p_graph.k());
    _round_moved_nodes = tbb::enumerable_thread_specific<std::vector<NodeID>>();
  }

  template <typename Lambda> void adjacent_nodes(const NodeID u, Lambda &&lambda) const {
    _graph->adjacent_nodes(u, std::forward<Lambda>(lambda));
  }

  [[nodiscard]] bool accept_neighbor(const NodeID u, const NodeID v) const {
    return _communities.empty() || _communities[u] == _communities[v];
  }

  [[nodiscard]] bool is_boundary_node(const PartitionedGraph &p_graph, const NodeID u) const {
    const BlockID from = p_graph.block(u);
    bool is_boundary = false;

    adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) {
      is_boundary |= accept_neighbor(u, v) && p_graph.block(v) != from;
    });

    return is_boundary;
  }

  NodeID initialize_active_nodes(const PartitionedGraph &p_graph) {
    tbb::enumerable_thread_specific<NodeID> active_nodes_ets(0);

    _graph->pfor_nodes([&](const NodeID u) {
      const std::uint8_t is_active = is_boundary_node(p_graph, u);
      _active[u] = is_active;
      _next_active[u] = 0;
      _moved[u] = 0;

      if (is_active) {
        ++active_nodes_ets.local();
      }
    });

    return active_nodes_ets.combine(std::plus{});
  }

  Gain compute_edge_cut(const PartitionedGraph &p_graph) {
    tbb::enumerable_thread_specific<Gain> cut_ets;

    _graph->pfor_nodes([&](const NodeID u) {
      auto &cut = cut_ets.local();
      const BlockID block_u = p_graph.block(u);
      adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        cut += (block_u != p_graph.block(v)) ? weight : 0;
      });
    });

    const Gain cut = cut_ets.combine(std::plus{});
    KASSERT(cut % 2 == 0, "inconsistent cut", assert::always);
    return cut / 2;
  }

  void clear_moved_nodes() {
    for (auto &nodes : _round_moved_nodes) {
      nodes.clear();
    }
  }

  NodeID perform_round(PartitionedGraph &p_graph) {
    tbb::enumerable_thread_specific<NodeID> num_moves_ets(0);
    const bool batch_block_weight_updates =
        tbb::this_task_arena::max_concurrency() >= 32 && p_graph.k() <= 256;

    _graph->pfor_nodes([&](const NodeID u) {
      if (!_active[u]) {
        return;
      }

      RatingMap &ratings = _rating_maps.local_ratings();
      ScalableVector<BlockID> &ties = _rating_maps.local_ties();
      const auto [to, expected_gain] = find_best_target(p_graph, u, ratings, ties);
      if (expected_gain <= 0 || to == p_graph.block(u)) {
        return;
      }

      const BlockID from = p_graph.block(u);
      if (batch_block_weight_updates) {
        p_graph.set_block<false>(u, to);
      } else {
        p_graph.set_block(u, to);
      }
      record_moved_node(u, from);
      ++num_moves_ets.local();
    });

    if (batch_block_weight_updates) {
      p_graph.recompute_block_weights();
    }
    return num_moves_ets.combine(std::plus{});
  }

  std::pair<BlockID, Gain> find_best_target(
      const PartitionedGraph &p_graph,
      const NodeID u,
      RatingMap &map,
      ScalableVector<BlockID> &tie_breaking_blocks
  ) {
    const std::size_t upper_bound_size = std::min<NodeID>(_graph->degree(u), p_graph.k());
    return map.execute(upper_bound_size, [&](auto &actual_map) {
      return find_best_target(p_graph, u, actual_map, tie_breaking_blocks);
    });
  }

  template <typename Map>
  std::pair<BlockID, Gain> find_best_target(
      const PartitionedGraph &p_graph,
      const NodeID u,
      Map &map,
      ScalableVector<BlockID> &tie_breaking_blocks
  ) {
    const BlockID from = p_graph.block(u);

    lp::NeighborhoodRatings::accumulate(
        *_graph,
        u,
        map,
        [&](const NodeID v) { return p_graph.block(v); },
        [&](const NodeID v) { return accept_neighbor(u, v); }
    );

    const auto selection = lp::select_best_positive_gain<Gain>(
        from, map.entries(), Random::instance(), tie_breaking_blocks
    );
    map.clear();
    return selection;
  }

  Gain compute_round_improvement(const PartitionedGraph &p_graph) const {
    tbb::enumerable_thread_specific<Gain> improvement_ets;

    tbb::parallel_for(_round_moved_nodes.range(), [&](const auto &range) {
      Gain &improvement = improvement_ets.local();
      for (const auto &nodes : range) {
        for (const NodeID u : nodes) {
          const BlockID old_u = _round_start_partition[u];
          const BlockID new_u = p_graph.block(u);

          adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
            const bool v_moved = __atomic_load_n(&_moved[v], __ATOMIC_RELAXED);
            // Count edges between two moved nodes once; moved-to-unmoved edges are seen once.
            if (v_moved && v < u) {
              return;
            }

            const BlockID old_v = v_moved ? _round_start_partition[v] : p_graph.block(v);
            const BlockID new_v = p_graph.block(v);

            improvement += (old_u != old_v ? weight : 0) - (new_u != new_v ? weight : 0);
          });
        }
      }
    });

    return improvement_ets.combine(std::plus{});
  }

  void record_moved_node(const NodeID u, const BlockID from) {
    std::uint8_t expected = 0;
    if (__atomic_compare_exchange_n(
            &_moved[u], &expected, 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
        )) {
      _round_start_partition[u] = from;
      _round_moved_nodes.local().push_back(u);
    }
  }

  void activate_moved_nodes() {
    tbb::parallel_for(_round_moved_nodes.range(), [&](const auto &range) {
      for (const auto &nodes : range) {
        for (const NodeID u : nodes) {
          __atomic_store_n(&_next_active[u], 1, __ATOMIC_RELAXED);
          adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) {
            if (accept_neighbor(u, v)) {
              __atomic_store_n(&_next_active[v], 1, __ATOMIC_RELAXED);
            }
          });
        }
      }
    });
  }

  NodeID update_active_nodes() {
    tbb::enumerable_thread_specific<NodeID> active_nodes_ets(0);

    _graph->pfor_nodes([&](const NodeID u) {
      const std::uint8_t active = __atomic_load_n(&_next_active[u], __ATOMIC_RELAXED);

      _active[u] = active;
      _next_active[u] = 0;
      _moved[u] = 0;

      if (active) {
        ++active_nodes_ets.local();
      }
    });

    return active_nodes_ets.combine(std::plus{});
  }

  void restore_partition(PartitionedGraph &p_graph) {
    tbb::parallel_for(_round_moved_nodes.range(), [&](const auto &range) {
      for (const auto &nodes : range) {
        for (const NodeID u : nodes) {
          const BlockID block = _round_start_partition[u];
          if (p_graph.block(u) != block) {
            p_graph.set_block(u, block);
          }
        }
      }
    });
  }

  const LabelPropagationRefinementContext &_lp_ctx;
  const Graph *_graph = nullptr;

  std::span<const NodeID> _communities;

  StaticArray<std::uint8_t> _active;
  StaticArray<std::uint8_t> _next_active;
  StaticArray<std::uint8_t> _moved;
  StaticArray<BlockID> _round_start_partition;

  RatingMaps _rating_maps;
  tbb::enumerable_thread_specific<std::vector<NodeID>> _round_moved_nodes;

  MultiQueueOverloadBalancer _balancer;
};

class UnconstrainedLPRefinerImplWrapper {
public:
  explicit UnconstrainedLPRefinerImplWrapper(const Context &ctx) : _ctx(ctx) {}

  void initialize(const PartitionedGraph &p_graph) {
    reified(p_graph, [&]<typename Graph>(const Graph &graph) { ensure_impl(graph); });
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("Unconstrained Label Propagation");

    return reified(p_graph, [&]<typename Graph>(const Graph &graph) {
      return ensure_impl(graph).refine(p_graph, p_ctx);
    });
  }

  void set_communities(std::span<const NodeID> communities) {
    _communities = communities;
    _impl.if_present([&](auto &impl) { impl.set_communities(_communities); });
  }

private:
  template <typename Graph> UnconstrainedLPRefinerImpl<Graph> &ensure_impl(const Graph &graph) {
    UnconstrainedLPRefinerImpl<Graph> &impl = _impl.ensure<Graph>(_ctx);
    impl.initialize(&graph);
    impl.set_communities(_communities);
    return impl;
  }

  const Context &_ctx;
  ReifiedGraphComponent<UnconstrainedLPRefinerImpl> _impl;
  std::span<const NodeID> _communities;
};

//
// Exposed wrapper
//

UnconstrainedLabelPropagationRefiner::UnconstrainedLabelPropagationRefiner(const Context &ctx)
    : _impl_wrapper(std::make_unique<UnconstrainedLPRefinerImplWrapper>(ctx)) {}

UnconstrainedLabelPropagationRefiner::~UnconstrainedLabelPropagationRefiner() = default;

std::string UnconstrainedLabelPropagationRefiner::name() const {
  return "Unconstrained Label Propagation";
}

void UnconstrainedLabelPropagationRefiner::initialize(const PartitionedGraph &p_graph) {
  _impl_wrapper->initialize(p_graph);
}

bool UnconstrainedLabelPropagationRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  if (p_ctx.has_min_block_weights()) {
    LOG_WARNING << "Unconstrained label propagation refinement does not support min block weights. "
                   "They will be ignored.";
  }

  return _impl_wrapper->refine(p_graph, p_ctx);
}

void UnconstrainedLabelPropagationRefiner::set_communities(std::span<const NodeID> communities) {
  _impl_wrapper->set_communities(communities);
}

} // namespace kaminpar::shm
