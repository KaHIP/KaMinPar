/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"
#include "kaminpar-shm/label_propagation.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename Graph>
struct LabelPropagationRefinerConfig : public LabelPropagationConfig<Graph> {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID, SparseMap<NodeID, EdgeWeight>>;
  static constexpr bool kUseHardWeightConstraint = true;
  static constexpr bool kReportEmptyClusters = false;
};

template <typename Graph>
class LabelPropagationRefinerImpl final : public ChunkRandomdLabelPropagation<
                                              LabelPropagationRefinerImpl<Graph>,
                                              LabelPropagationRefinerConfig,
                                              Graph> {
  using Base = ChunkRandomdLabelPropagation<
      LabelPropagationRefinerImpl<Graph>,
      LabelPropagationRefinerConfig,
      Graph>;
  friend Base;

  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  LabelPropagationRefinerImpl(const Context &ctx)
      : _r_ctx{ctx.refinement},
        _n(ctx.partition.n),
        _k(ctx.partition.k) {
    Base::allocate(ctx.partition.n, ctx.partition.n, ctx.partition.k, false);
    this->set_max_degree(_r_ctx.lp.large_degree_threshold);
    this->set_max_num_neighbors(_r_ctx.lp.max_num_neighbors);
  }

  void initialize(const Graph *graph) {
    _graph = graph;
  }

  void allocate() {
    SCOPED_HEAP_PROFILER("Label Propagation Allocation");
    Base::allocate(_n, _n, _k, true);
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    KASSERT(_graph == &p_graph.graph());
    KASSERT(p_graph.k() <= p_ctx.k);
    _p_graph = &p_graph;
    _p_ctx = &p_ctx;
    Base::initialize(_graph, _p_ctx->k);

    SCOPED_TIMER("Label Propagation");
    const std::size_t max_iterations =
        _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      if (this->perform_iteration() == 0) {
        return false;
      }
    }

    return true;
  }

  using Base::expected_total_gain;

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

  bool accept_cluster(const typename Base::ClusterSelectionState &state) {
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
            current_overload < initial_overload || state.current_cluster == state.initial_cluster);
  }

  const NodeID _n;
  const NodeID _k;
  const Graph *_graph{nullptr};
  PartitionedGraph *_p_graph{nullptr};
  const PartitionContext *_p_ctx;
  const RefinementContext &_r_ctx;
};

class LabelPropagationRefiner : public Refiner {
public:
  LabelPropagationRefiner(const Context &ctx);

  ~LabelPropagationRefiner() override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  std::unique_ptr<LabelPropagationRefinerImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LabelPropagationRefinerImpl<CompactCSRGraph>> _compact_csr_impl;
  std::unique_ptr<LabelPropagationRefinerImpl<CompressedGraph>> _compressed_impl;

  // The data structures which are used by the LP clustering and are shared between the
  // different graph implementations.
  bool _allocated = false;
  LabelPropagationRefinerImpl<Graph>::DataStructures _structs;
};
} // namespace kaminpar::shm
