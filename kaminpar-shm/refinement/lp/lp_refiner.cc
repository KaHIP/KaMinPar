/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

#include "kaminpar-shm/label_propagation.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/heap_profiler.h"
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
        return false;
      }
    }

    return true;
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

  bool move_cluster_weight(
      const BlockID old_block,
      const BlockID new_block,
      const BlockWeight delta,
      const BlockWeight max_weight
  ) {
    return _p_graph->move_block_weight(old_block, new_block, delta, max_weight);
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
    return _p_ctx->block_weights.max(block);
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

          if (state.current_cluster_weight + state.u_weight < current_max_weight ||
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

            if (state.current_cluster_weight + state.u_weight < current_max_weight ||
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

            if (state.current_cluster_weight + state.u_weight < current_max_weight ||
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
               (state.current_cluster_weight + state.u_weight < current_max_weight ||
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
};

class LPRefinerImplWrapper {
public:
  LPRefinerImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<LPRefinerImpl<CSRGraph>>(ctx, _permutations)),
        _compressed_impl(std::make_unique<LPRefinerImpl<CompressedGraph>>(ctx, _permutations)) {}

  void initialize(const PartitionedGraph &p_graph) {
    p_graph.graph().reified(
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

    return p_graph.graph().reified(
        [&](const auto &) { return refine(*_csr_impl); },
        [&](const auto &) { return refine(*_compressed_impl); }
    );
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

} // namespace kaminpar::shm
