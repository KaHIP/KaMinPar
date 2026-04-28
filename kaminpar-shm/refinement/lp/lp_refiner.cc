/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

#include "kaminpar-common/algorithms/label_propagation.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/iteration.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Actual implementation -- not exposed in header
//

namespace {

constexpr NodeID kMinChunkSize = 1024;
constexpr NodeID kPermutationSize = 64;
constexpr std::size_t kNumberOfNodePermutations = 64;
constexpr std::size_t kRatingMapThreshold = 10000;

using LPRefinerRatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID, rm_backyard::SparseMap>;
using LPRefinerGrowingRatingMap = DynamicRememberingFlatMap<BlockID, EdgeWeight>;
using LPRefinerConcurrentRatingMap = ConcurrentFastResetArray<EdgeWeight, BlockID>;
using LPRefinerWorkspace = lp::Workspace<
    NodeID,
    BlockID,
    EdgeWeight,
    LPRefinerRatingMap,
    LPRefinerGrowingRatingMap,
    LPRefinerConcurrentRatingMap,
    true>;
using LPRefinerOrderWorkspace =
    iteration::ChunkRandomNodeOrderWorkspace<NodeID, kPermutationSize, kNumberOfNodePermutations>;

lp::RatingMapStrategy map_rating_map_strategy(const LabelPropagationImplementation impl) {
  switch (impl) {
  case LabelPropagationImplementation::SINGLE_PHASE:
    return lp::RatingMapStrategy::SINGLE_PHASE;
  case LabelPropagationImplementation::TWO_PHASE:
    return lp::RatingMapStrategy::TWO_PHASE;
  case LabelPropagationImplementation::GROWING_HASH_TABLES:
    return lp::RatingMapStrategy::GROWING_HASH_TABLES;
  }
  __builtin_unreachable();
}

lp::TieBreakingStrategy map_tie_breaking_strategy(const TieBreakingStrategy strategy) {
  switch (strategy) {
  case TieBreakingStrategy::GEOMETRIC:
    return lp::TieBreakingStrategy::GEOMETRIC;
  case TieBreakingStrategy::UNIFORM:
    return lp::TieBreakingStrategy::UNIFORM;
  }
  __builtin_unreachable();
}

class PartitionedGraphLabelStore {
public:
  using ClusterIDType = BlockID;

  void init(PartitionedGraph &p_graph) {
    _p_graph = &p_graph;
  }

  void init_cluster(const NodeID, const BlockID) {}

  [[nodiscard]] BlockID initial_cluster(const NodeID u) const {
    return _p_graph->block(u);
  }

  [[nodiscard]] BlockID cluster(const NodeID u) const {
    return _p_graph->block(u);
  }

  void move_node(const NodeID u, const BlockID block) {
    _p_graph->set_block<false>(u, block);
  }

private:
  PartitionedGraph *_p_graph = nullptr;
};

class PartitionedGraphWeightStore {
public:
  using ClusterWeightType = BlockWeight;

  void init(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _p_ctx = &p_ctx;
  }

  void init_cluster_weight(const BlockID, const BlockWeight) {}

  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) const {
    return _p_graph->block_weight(b);
  }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID b) const {
    return _p_graph->block_weight(b);
  }

  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID b) const {
    return _p_ctx->max_block_weight(b);
  }

  [[nodiscard]] BlockWeight min_cluster_weight(const BlockID b) const {
    return _p_ctx->min_block_weight(b);
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

  void reassign_cluster_weights(const StaticArray<BlockID> &, BlockID) {}

private:
  PartitionedGraph *_p_graph = nullptr;
  const PartitionContext *_p_ctx = nullptr;
};

struct LPRefinerNeighborPolicy {
  std::span<const NodeID> communities;

  [[nodiscard]] bool accept(const NodeID u, const NodeID v) const {
    return communities.empty() || communities[u] == communities[v];
  }

  [[nodiscard]] bool activate(const NodeID) const {
    return true;
  }

  [[nodiscard]] bool skip(const NodeID) const {
    return false;
  }
};

class LPRefinerSelector {
public:
  LPRefinerSelector(PartitionedGraphWeightStore &weights, const RefinementContext &r_ctx)
      : _weights(weights),
        _r_ctx(r_ctx) {}

  template <typename State, typename RatingMap>
  [[nodiscard]] BlockID select(
      const bool store_favored_cluster,
      const EdgeWeight gain_delta,
      State &state,
      RatingMap &map,
      ScalableVector<BlockID> &tie_breaking_clusters,
      ScalableVector<BlockID> &tie_breaking_favored_clusters
  ) {
    if (state.initial_cluster_weight - state.u_weight <
        _weights.min_cluster_weight(state.initial_cluster)) {
      return state.initial_cluster;
    }

    const bool use_uniform_tie_breaking =
        _r_ctx.lp.tie_breaking_strategy == TieBreakingStrategy::UNIFORM;

    BlockID favored_cluster = state.initial_cluster;
    if (use_uniform_tie_breaking) {
      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = _weights.cluster_weight(cluster);

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
          const NodeWeight current_max_weight = _weights.max_cluster_weight(state.current_cluster);
          const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
          const NodeWeight initial_overload =
              state.initial_cluster_weight - _weights.max_cluster_weight(state.initial_cluster);

          if (state.current_cluster_weight + state.u_weight <= current_max_weight ||
              current_overload < initial_overload ||
              state.current_cluster == state.initial_cluster) {
            tie_breaking_clusters.clear();
            tie_breaking_clusters.push_back(state.current_cluster);

            state.best_cluster = state.current_cluster;
            state.best_cluster_weight = state.current_cluster_weight;
            state.best_gain = state.current_gain;
          }
        } else if (state.current_gain == state.best_gain) {
          const NodeWeight current_max_weight = _weights.max_cluster_weight(state.current_cluster);
          const NodeWeight best_overload =
              state.best_cluster_weight - _weights.max_cluster_weight(state.best_cluster);
          const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;

          if (current_overload < best_overload) {
            const NodeWeight initial_overload =
                state.initial_cluster_weight - _weights.max_cluster_weight(state.initial_cluster);

            if (state.current_cluster_weight + state.u_weight <= current_max_weight ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster) {
              tie_breaking_clusters.clear();
              tie_breaking_clusters.push_back(state.current_cluster);

              state.best_cluster = state.current_cluster;
              state.best_cluster_weight = state.current_cluster_weight;
            }
          } else if (current_overload == best_overload) {
            const NodeWeight initial_overload =
                state.initial_cluster_weight - _weights.max_cluster_weight(state.initial_cluster);

            if (state.current_cluster_weight + state.u_weight <= current_max_weight ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster) {
              tie_breaking_clusters.push_back(state.current_cluster);
            }
          }
        }
      }

      if (tie_breaking_clusters.size() > 1) {
        const BlockID i = state.local_rand.random_index(0, tie_breaking_clusters.size());
        state.best_cluster = tie_breaking_clusters[i];
      }
      tie_breaking_clusters.clear();

      if (tie_breaking_favored_clusters.size() > 1) {
        const BlockID i = state.local_rand.random_index(0, tie_breaking_favored_clusters.size());
        favored_cluster = tie_breaking_favored_clusters[i];
      }
      tie_breaking_favored_clusters.clear();

      return favored_cluster;
    } else {
      const auto accept_cluster = [&] {
        static_assert(std::is_signed_v<NodeWeight>);

        const NodeWeight current_max_weight = _weights.max_cluster_weight(state.current_cluster);
        const NodeWeight best_overload =
            state.best_cluster_weight - _weights.max_cluster_weight(state.best_cluster);
        const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
        const NodeWeight initial_overload =
            state.initial_cluster_weight - _weights.max_cluster_weight(state.initial_cluster);

        return (state.current_gain > state.best_gain ||
                (state.current_gain == state.best_gain &&
                 (current_overload < best_overload ||
                  (current_overload == best_overload && state.local_rand.random_bool())))) &&
               (state.current_cluster_weight + state.u_weight <= current_max_weight ||
                current_overload < initial_overload ||
                state.current_cluster == state.initial_cluster);
      };

      for (const auto [cluster, rating] : map.entries()) {
        state.current_cluster = cluster;
        state.current_gain = rating - gain_delta;
        state.current_cluster_weight = _weights.cluster_weight(cluster);

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

private:
  PartitionedGraphWeightStore &_weights;
  const RefinementContext &_r_ctx;
};

} // namespace

template <typename Graph> class LPRefinerImpl final {
  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

  SET_DEBUG(true);

public:
  LPRefinerImpl(
      const Context &ctx, LPRefinerWorkspace &workspace, LPRefinerOrderWorkspace &order_workspace
  )
      : _r_ctx(ctx.refinement),
        _workspace(workspace),
        _order_workspace(order_workspace),
        _selector(_weights, _r_ctx) {}

  void allocate() {
    SCOPED_HEAP_PROFILER("Allocation");
    SCOPED_TIMER("Allocation");
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

    _labels.init(p_graph);
    _weights.init(p_graph, p_ctx);
    _order_workspace.clear_order();

    lp::Options<NodeID, BlockID> options{
        .max_degree = _r_ctx.lp.large_degree_threshold,
        .max_num_neighbors = _r_ctx.lp.max_num_neighbors,
        .desired_num_clusters = 0,
        .rating_map_strategy = map_rating_map_strategy(_r_ctx.lp.impl),
        .active_set_strategy = lp::ActiveSetStrategy::GLOBAL,
        .tie_breaking_strategy = map_tie_breaking_strategy(_r_ctx.lp.tie_breaking_strategy),
        .track_cluster_count = false,
        .use_two_hop_clustering = false,
        .use_actual_gain = false,
        .relabel_before_second_phase = false,
        .rating_map_threshold = kRatingMapThreshold,
    };
    LPRefinerNeighborPolicy neighbors{.communities = _communities};
    lp::LabelPropagationCore core(
        *_graph, _labels, _weights, _selector, neighbors, _workspace, options
    );
    core.initialize(
        {.num_nodes = _graph->n(), .num_active_nodes = _graph->n(), .num_clusters = _p_ctx->k}
    );

    const std::size_t max_iterations =
        _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));

      iteration::ChunkRandomNodeOrder order(
          *_graph,
          _order_workspace,
          iteration::NodeRange<NodeID>{0, _graph->n()},
          static_cast<EdgeID>(kMinChunkSize),
          iteration::bucket_limit_for_max_degree(*_graph, options.max_degree)
      );
      const auto result = lp::run_iteration(order, core);
      if (result.moved_nodes == 0) {
        break;
      }
    }

    return true;
  }

  void set_communities(std::span<const NodeID> communities) {
    _communities = communities;
  }

  const Graph *_graph = nullptr;
  PartitionedGraph *_p_graph = nullptr;

  const PartitionContext *_p_ctx;
  const RefinementContext &_r_ctx;
  LPRefinerWorkspace &_workspace;
  LPRefinerOrderWorkspace &_order_workspace;
  PartitionedGraphLabelStore _labels;
  PartitionedGraphWeightStore _weights;
  LPRefinerSelector _selector;

  std::span<const NodeID> _communities;
};

class LPRefinerImplWrapper {
public:
  LPRefinerImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<LPRefinerImpl<CSRGraph>>(ctx, _workspace, _order_workspace)),
        _compressed_impl(
            std::make_unique<LPRefinerImpl<CompressedGraph>>(ctx, _workspace, _order_workspace)
        ) {}

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
      }

      const bool found_improvement = impl.refine(p_graph, p_ctx);
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

  bool _freed = true;
  LPRefinerWorkspace _workspace;
  LPRefinerOrderWorkspace _order_workspace;
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

} // namespace kaminpar::shm
