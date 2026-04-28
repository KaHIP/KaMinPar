/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/label_propagation.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Actual implementation -- not exposed in header
//

struct LPRefinerConfig : public LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID, rm_backyard::SparseMap>;
  static constexpr bool kUseHardWeightConstraint = true;
  static constexpr bool kReportEmptyClusters = false;
};

template <typename Graph>
class LPRefinerImpl final
    : public ChunkRandomLabelPropagation<LPRefinerImpl<Graph>, LPRefinerConfig, Graph> {
  using Base = ChunkRandomLabelPropagation<LPRefinerImpl<Graph>, LPRefinerConfig, Graph>;
  friend Base;

  using Config = LPRefinerConfig;
  using ClusterID = Config::ClusterID;

  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

  SET_DEBUG(true);

public:
  using Permutations = Base::Permutations;

  LPRefinerImpl(const Context &ctx, Permutations &permutations)
      : Base(permutations),
        _r_ctx(ctx.refinement) {
    Base::preinitialize(ctx.partition.n, ctx.partition.k);
    Base::set_max_degree(_r_ctx.lp.large_degree_threshold);
    Base::set_max_num_neighbors(_r_ctx.lp.max_num_neighbors);
    Base::set_implementation(_r_ctx.lp.impl);
    Base::set_tie_breaking_strategy(_r_ctx.lp.tie_breaking_strategy);
    Base::set_relabel_before_second_phase(false);
  }

  void allocate() {
    SCOPED_HEAP_PROFILER("Allocation");
    SCOPED_TIMER("Allocation");

    Base::allocate();
  }

  void initialize(const Graph *graph) {
    _graph = graph;
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    KASSERT(_graph == p_graph.graph().underlying_graph());
    KASSERT(p_graph.k() <= p_ctx.k);
    SCOPED_HEAP_PROFILER("Label Propagation");

    _p_graph = &p_graph;
    _p_ctx = &p_ctx;

    Base::initialize(_graph, _p_ctx->k);

    const std::size_t max_iterations =
        _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));

      if (Base::perform_iteration() == 0) {
        break;
      }
    }

    return true;
  }

  void set_communities(std::span<const NodeID> communities) {
    _communities = communities;
  }

public:
  [[nodiscard]] BlockID initial_cluster(const NodeID u) {
    return _p_graph->block(u);
  }

  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) {
    return _p_graph->block_weight(b);
  }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID b) {
    return _p_graph->block_weight(b);
  }

  [[nodiscard]] bool accept_neighbor(const NodeID u, const NodeID v) {
    return _communities.empty() || _communities[u] == _communities[v];
  }

  bool move_cluster_weight(
      const BlockID old_block,
      const BlockID new_block,
      const BlockWeight delta,
      const BlockWeight max_weight
  ) {
    return _p_graph->move_block_weight(
        old_block, new_block, delta, max_weight, min_cluster_weight(old_block)
    );
  }

  void reassign_cluster_weights(
      const StaticArray<BlockID> & /* mapping */, const BlockID /* num_new_clusters */
  ) {}

  void init_cluster(const NodeID /* u */, const BlockID /* b */) {}

  void init_cluster_weight(const BlockID /* b */, const BlockWeight /* weight */) {}

  [[nodiscard]] BlockID cluster(const NodeID u) {
    return _p_graph->block(u);
  }

  void move_node(const NodeID u, const BlockID block) {
    _p_graph->set_block<false>(u, block);
  }

  [[nodiscard]] BlockID num_clusters() {
    return _p_graph->k();
  }

  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID block) {
    return _p_ctx->max_block_weight(block);
  }

  [[nodiscard]] BlockWeight min_cluster_weight(const BlockID block) const {
    return _p_ctx->min_block_weight(block);
  }

  template <typename RatingMap>
  [[nodiscard]] ClusterID select_best_cluster(
      const bool store_favored_cluster,
      const EdgeWeight gain_delta,
      Base::ClusterSelectionState &state,
      RatingMap &map,
      ScalableVector<ClusterID> &tie_breaking_clusters,
      ScalableVector<ClusterID> &tie_breaking_favored_clusters
  ) {
    if (state.initial_cluster_weight - state.u_weight < min_cluster_weight(state.initial_cluster)) {
      return state.initial_cluster;
    }

    const bool use_uniform_tie_breaking = _tie_breaking_strategy == TieBreakingStrategy::UNIFORM;

    ClusterID favored_cluster = state.initial_cluster;
    if (use_uniform_tie_breaking) {
      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = cluster_weight(cluster);

        if (store_favored_cluster) {
          if (state.current_gain > state.overall_best_gain) {
            state.overall_best_gain = state.current_gain;
            favored_cluster = state.current_cluster;

            tie_breaking_favored_clusters.clear();
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          } else if (state.current_gain == state.overall_best_gain) {
            tie_breaking_favored_clusters.push_back(state.current_cluster);
          }
        }

        if (state.current_gain > state.best_gain) {
          const NodeWeight current_max_weight = max_cluster_weight(state.current_cluster);
          const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
          const NodeWeight initial_overload =
              state.initial_cluster_weight - max_cluster_weight(state.initial_cluster);

          if (((state.current_cluster_weight + state.u_weight <= current_max_weight)) ||
              current_overload < initial_overload ||
              state.current_cluster == state.initial_cluster) {
            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(state.current_cluster);

            state.best_cluster = state.current_cluster;
            state.best_cluster_weight = state.current_cluster_weight;
            state.best_gain = state.current_gain;
          }
        } else if (state.current_gain == state.best_gain) {
          const NodeWeight current_max_weight = max_cluster_weight(state.current_cluster);
          const NodeWeight best_overload =
              state.best_cluster_weight - max_cluster_weight(state.best_cluster);
          const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;

          if (current_overload < best_overload) {
            const NodeWeight initial_overload =
                state.initial_cluster_weight - max_cluster_weight(state.initial_cluster);

            if (((state.current_cluster_weight + state.u_weight <= current_max_weight)) ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster) {
              tie_breaking_clusters.clear();
              tie_breaking_clusters.push_back(state.current_cluster);

              state.best_cluster = state.current_cluster;
              state.best_cluster_weight = state.current_cluster_weight;
            }
          } else if (current_overload == best_overload) {
            const NodeWeight initial_overload =
                state.initial_cluster_weight - max_cluster_weight(state.initial_cluster);

            if (state.current_cluster_weight + state.u_weight <= current_max_weight ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster) {
              tie_breaking_clusters.push_back(state.current_cluster);
            }
          }
        }
      }

      if (tie_breaking_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_clusters.size());
        state.best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const ClusterID i = state.local_rand.random_index(0, tie_breaking_favored_clusters.size());
        favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();

      return favored_cluster;
    } else {
      const auto accept_cluster = [&] {
        static_assert(std::is_signed_v<NodeWeight>);

        const NodeWeight current_max_weight = max_cluster_weight(state.current_cluster);
        const NodeWeight best_overload =
            state.best_cluster_weight - max_cluster_weight(state.best_cluster);
        const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
        const NodeWeight initial_overload =
            state.initial_cluster_weight - max_cluster_weight(state.initial_cluster);

        return (state.current_gain > state.best_gain ||
                (state.current_gain == state.best_gain &&
                 (current_overload < best_overload ||
                  (current_overload == best_overload && state.local_rand.random_bool())))) &&
               (((state.current_cluster_weight + state.u_weight <= current_max_weight)) ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster);
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = cluster_weight(cluster);

        if (store_favored_cluster && state.current_gain > state.overall_best_gain) {
          state.overall_best_gain = state.current_gain;
          favored_cluster = state.current_cluster;
        }

        if (accept_cluster()) {
          state.best_cluster = state.current_cluster;
          state.best_cluster_weight = state.current_cluster_weight;
          state.best_gain = state.current_gain;
        }
      }

      return favored_cluster;
    }
  }

  using Base::_tie_breaking_strategy;
  using Base::expected_total_gain;

  const Graph *_graph = nullptr;
  PartitionedGraph *_p_graph = nullptr;

  const PartitionContext *_p_ctx;
  const RefinementContext &_r_ctx;

  std::span<const NodeID> _communities;
};

class LPRefinerImplWrapper {
public:
  LPRefinerImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<LPRefinerImpl<CSRGraph>>(ctx, _permutations)),
        _compressed_impl(std::make_unique<LPRefinerImpl<CompressedGraph>>(ctx, _permutations)) {}

  void initialize(const PartitionedGraph &p_graph) {
    reified(
        p_graph,
        [&](const auto &graph) { _csr_impl->initialize(&graph); },
        [&](const auto &graph) { _compressed_impl->initialize(&graph); }
    );
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("Label Propagation");

    const auto refine = [&](auto &impl) {
      if (_freed) {
        _freed = false;
        impl.allocate();
      } else {
        impl.setup(std::move(_structs));
      }

      const bool found_improvement = impl.refine(p_graph, p_ctx);

      _structs = impl.release();
      return found_improvement;
    };

    return reified(
        p_graph,
        [&](const auto &) { return refine(*_csr_impl); },
        [&](const auto &) { return refine(*_compressed_impl); }
    );
  }

  void set_communities(std::span<const NodeID> communities) {
    _csr_impl->set_communities(communities);
    _compressed_impl->set_communities(communities);
  }

private:
  std::unique_ptr<LPRefinerImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LPRefinerImpl<CompressedGraph>> _compressed_impl;

  // The data structures which are used by the LP refiner and are shared between the
  // different implementations.
  bool _freed = true;
  LPRefinerImpl<Graph>::Permutations _permutations;
  LPRefinerImpl<Graph>::DataStructures _structs;
};

template <typename Graph> class UnconstrainedLPRefinerImpl {
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID, rm_backyard::SparseMap>;

  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  explicit UnconstrainedLPRefinerImpl(const Context &ctx)
      : _r_ctx(ctx.refinement),
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
    _balancer.track_moves([&](const NodeID u, const BlockID /* from */, const BlockID /* to */) {
      mark_moved_and_activate_neighbors(u);
    });

    EdgeWeight cut_before = metrics::edge_cut(p_graph);
    NodeID active_nodes = initialize_active_nodes(p_graph);
    bool found_improvement = false;

    const std::size_t max_iterations =
        _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));

      if (active_nodes == 0) {
        break;
      }

      save_partition(p_graph);
      reset_round_state();

      const NodeID num_moves = perform_round(p_graph);
      if (num_moves == 0) {
        break;
      }

      if (metrics::total_overload(p_graph, p_ctx) > 0) {
        TIMED_SCOPE("Rebalance") {
          _balancer.refine(p_graph, p_ctx);
        };
      }

      const EdgeWeight cut_after = metrics::edge_cut(p_graph);
      if (metrics::total_overload(p_graph, p_ctx) > 0 || cut_after >= cut_before) {
        restore_partition(p_graph);
        break;
      }

      const EdgeWeight improvement = cut_before - cut_after;
      const double relative_improvement = cut_before == 0 ? 0.0 : 1.0 * improvement / cut_before;

      found_improvement = true;
      cut_before = cut_after;
      active_nodes = update_active_nodes();

      if (relative_improvement < _r_ctx.lp.unconstrained_min_improvement_factor) {
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

    _rating_maps =
        tbb::enumerable_thread_specific<RatingMap>([&] { return RatingMap(p_graph.k()); });
    _tie_breaking_blocks = tbb::enumerable_thread_specific<std::vector<BlockID>>();
  }

  [[nodiscard]] bool should_handle_node(const NodeID u) const {
    return _graph->degree(u) <= _r_ctx.lp.large_degree_threshold;
  }

  template <typename Lambda> void adjacent_nodes(const NodeID u, Lambda &&lambda) const {
    if (_r_ctx.lp.max_num_neighbors == std::numeric_limits<NodeID>::max()) {
      _graph->adjacent_nodes(u, std::forward<Lambda>(lambda));
    } else {
      _graph->adjacent_nodes(u, _r_ctx.lp.max_num_neighbors, std::forward<Lambda>(lambda));
    }
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
    std::atomic<NodeID> active_nodes = 0;

    _graph->pfor_nodes([&](const NodeID u) {
      const std::uint8_t is_active = should_handle_node(u) && is_boundary_node(p_graph, u);
      _active[u] = is_active;
      _next_active[u] = 0;
      _moved[u] = 0;

      if (is_active) {
        active_nodes.fetch_add(1, std::memory_order_relaxed);
      }
    });

    return active_nodes.load(std::memory_order_relaxed);
  }

  void save_partition(const PartitionedGraph &p_graph) {
    _graph->pfor_nodes([&](const NodeID u) { _round_start_partition[u] = p_graph.block(u); });
  }

  void reset_round_state() {
    _graph->pfor_nodes([&](const NodeID u) {
      _next_active[u] = 0;
      _moved[u] = 0;
    });
  }

  NodeID perform_round(PartitionedGraph &p_graph) {
    std::atomic<NodeID> num_moves = 0;

    _graph->pfor_nodes([&](const NodeID u) {
      if (!_active[u] || !should_handle_node(u)) {
        return;
      }

      const auto [to, gain] =
          find_best_target(p_graph, u, _rating_maps.local(), _tie_breaking_blocks.local());
      if (gain <= 0 || to == p_graph.block(u)) {
        return;
      }

      p_graph.set_block(u, to);
      mark_moved_and_activate_neighbors(u);
      num_moves.fetch_add(1, std::memory_order_relaxed);
    });

    return num_moves.load(std::memory_order_relaxed);
  }

  std::pair<BlockID, EdgeWeight> find_best_target(
      const PartitionedGraph &p_graph,
      const NodeID u,
      RatingMap &map,
      std::vector<BlockID> &tie_breaking_blocks
  ) {
    const std::size_t upper_bound_size = std::min<NodeID>(_graph->degree(u), p_graph.k());
    return map.execute(upper_bound_size, [&](auto &actual_map) {
      return find_best_target(p_graph, u, actual_map, tie_breaking_blocks);
    });
  }

  template <typename Map>
  std::pair<BlockID, EdgeWeight> find_best_target(
      const PartitionedGraph &p_graph,
      const NodeID u,
      Map &map,
      std::vector<BlockID> &tie_breaking_blocks
  ) {
    const BlockID from = p_graph.block(u);

    adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (accept_neighbor(u, v)) {
        map[p_graph.block(v)] += weight;
      }
    });

    const EdgeWeight gain_delta = map[from];
    BlockID best_block = from;
    EdgeWeight best_gain = 0;

    const bool uniform_tie_breaking =
        _r_ctx.lp.tie_breaking_strategy == TieBreakingStrategy::UNIFORM;
    if (uniform_tie_breaking) {
      tie_breaking_blocks.clear();
    }

    for (const auto [block, rating] : map.entries()) {
      if (block == from) {
        continue;
      }

      const EdgeWeight gain = rating - gain_delta;
      if (gain > best_gain) {
        best_block = block;
        best_gain = gain;

        if (uniform_tie_breaking) {
          tie_breaking_blocks.clear();
          tie_breaking_blocks.push_back(block);
        }
      } else if (uniform_tie_breaking && gain == best_gain && gain > 0) {
        tie_breaking_blocks.push_back(block);
      }
    }

    if (uniform_tie_breaking && tie_breaking_blocks.size() > 1) {
      const std::size_t i = Random::instance().random_index(0, tie_breaking_blocks.size());
      best_block = tie_breaking_blocks[i];
    }

    map.clear();
    return {best_block, best_gain};
  }

  void mark_moved_and_activate_neighbors(const NodeID u) {
    __atomic_store_n(&_moved[u], 1, __ATOMIC_RELAXED);

    adjacent_nodes(u, [&](const NodeID v, const EdgeWeight) {
      if (accept_neighbor(u, v)) {
        __atomic_store_n(&_next_active[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

  NodeID update_active_nodes() {
    std::atomic<NodeID> active_nodes = 0;

    _graph->pfor_nodes([&](const NodeID u) {
      const std::uint8_t active = should_handle_node(u) &&
                                  !__atomic_load_n(&_moved[u], __ATOMIC_RELAXED) &&
                                  __atomic_load_n(&_next_active[u], __ATOMIC_RELAXED);

      _active[u] = active;
      _next_active[u] = 0;
      _moved[u] = 0;

      if (active) {
        active_nodes.fetch_add(1, std::memory_order_relaxed);
      }
    });

    return active_nodes.load(std::memory_order_relaxed);
  }

  void restore_partition(PartitionedGraph &p_graph) {
    _graph->pfor_nodes([&](const NodeID u) {
      const BlockID block = _round_start_partition[u];
      if (p_graph.block(u) != block) {
        p_graph.set_block(u, block);
      }
    });
  }

  const RefinementContext &_r_ctx;
  const Graph *_graph = nullptr;

  std::span<const NodeID> _communities;

  StaticArray<std::uint8_t> _active;
  StaticArray<std::uint8_t> _next_active;
  StaticArray<std::uint8_t> _moved;
  StaticArray<BlockID> _round_start_partition;

  tbb::enumerable_thread_specific<RatingMap> _rating_maps;
  tbb::enumerable_thread_specific<std::vector<BlockID>> _tie_breaking_blocks;

  OverloadBalancer _balancer;
};

class UnconstrainedLPRefinerImplWrapper {
public:
  explicit UnconstrainedLPRefinerImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<UnconstrainedLPRefinerImpl<CSRGraph>>(ctx)),
        _compressed_impl(std::make_unique<UnconstrainedLPRefinerImpl<CompressedGraph>>(ctx)) {}

  void initialize(const PartitionedGraph &p_graph) {
    reified(
        p_graph,
        [&](const auto &graph) { _csr_impl->initialize(&graph); },
        [&](const auto &graph) { _compressed_impl->initialize(&graph); }
    );
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("Unconstrained Label Propagation");

    return reified(
        p_graph,
        [&](const auto &) { return _csr_impl->refine(p_graph, p_ctx); },
        [&](const auto &) { return _compressed_impl->refine(p_graph, p_ctx); }
    );
  }

  void set_communities(std::span<const NodeID> communities) {
    _csr_impl->set_communities(communities);
    _compressed_impl->set_communities(communities);
  }

private:
  std::unique_ptr<UnconstrainedLPRefinerImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<UnconstrainedLPRefinerImpl<CompressedGraph>> _compressed_impl;
};

//
// Exposed wrapper
//

LabelPropagationRefiner::LabelPropagationRefiner(const Context &ctx)
    : _impl_wrapper(std::make_unique<LPRefinerImplWrapper>(ctx)) {}

LabelPropagationRefiner::~LabelPropagationRefiner() = default;

std::string LabelPropagationRefiner::name() const {
  return "Label Propagation";
}

void LabelPropagationRefiner::initialize(const PartitionedGraph &p_graph) {
  _impl_wrapper->initialize(p_graph);
}

bool LabelPropagationRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return _impl_wrapper->refine(p_graph, p_ctx);
}

void LabelPropagationRefiner::set_communities(std::span<const NodeID> communities) {
  _impl_wrapper->set_communities(communities);
}

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
