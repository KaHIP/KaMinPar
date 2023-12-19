/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

#include "kaminpar-shm/label_propagation.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
//
// Private implementation
//

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
  LabelPropagationRefinerImpl(const Context &ctx) : _r_ctx{ctx.refinement} {
    this->allocate(ctx.partition.n, ctx.partition.n, ctx.partition.k);
    this->set_max_degree(_r_ctx.lp.large_degree_threshold);
    this->set_max_num_neighbors(_r_ctx.lp.max_num_neighbors);
  }

  void initialize(const Graph *graph) {
    _graph = graph;
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

  const Graph *_graph{nullptr};
  PartitionedGraph *_p_graph{nullptr};
  const PartitionContext *_p_ctx;
  const RefinementContext &_r_ctx;
};

//
// Exposed wrapper
//

LabelPropagationRefiner::LabelPropagationRefiner(const Context &ctx)
    : _csr_impl{std::make_unique<LabelPropagationRefinerImpl<CSRGraph>>(ctx)},
      _compressed_impl{std::make_unique<LabelPropagationRefinerImpl<CompressedGraph>>(ctx)} {}

LabelPropagationRefiner::~LabelPropagationRefiner() = default;

void LabelPropagationRefiner::initialize(const PartitionedGraph &p_graph) {
  if (auto *csr_graph = dynamic_cast<CSRGraph *>(p_graph.graph().underlying_graph());
      csr_graph != nullptr) {
    _csr_impl->initialize(csr_graph);
    return;
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(p_graph.graph().underlying_graph());
      compressed_graph != nullptr) {
    _compressed_impl->initialize(compressed_graph);
    return;
  }

  __builtin_unreachable();
}

bool LabelPropagationRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  if (auto *csr_graph = dynamic_cast<CSRGraph *>(p_graph.graph().underlying_graph());
      csr_graph != nullptr) {
    return _csr_impl->refine(p_graph, p_ctx);
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(p_graph.graph().underlying_graph());
      compressed_graph != nullptr) {
    return _compressed_impl->refine(p_graph, p_ctx);
  }

  __builtin_unreachable();
}
} // namespace kaminpar::shm
