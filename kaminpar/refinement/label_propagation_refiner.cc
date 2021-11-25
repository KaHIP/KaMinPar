/*******************************************************************************
 * @file:   label_propagation_refiner.cc
 *
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:  Label propagation refinement graphutils.
 ******************************************************************************/
#include "kaminpar/refinement/label_propagation_refiner.h"

#include "kaminpar/label_propagation.h"
#include "kaminpar/utility/timer.h"

namespace kaminpar {
//
// Private implementation
//

struct LabelPropagationRefinerConfig : public LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, SparseMap<NodeID, EdgeWeight>>;
  static constexpr bool kUseHardWeightConstraint = true;
  static constexpr bool kReportEmptyClusters = false;
};

class LabelPropagationRefinerImpl final
    : public ChunkRandomizedLabelPropagation<LabelPropagationRefinerImpl, LabelPropagationRefinerConfig> {
  using Base = ChunkRandomizedLabelPropagation<LabelPropagationRefinerImpl, LabelPropagationRefinerConfig>;
  friend Base;

  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  LabelPropagationRefinerImpl(const Graph &graph, const RefinementContext &r_ctx) : _r_ctx{r_ctx} {
    allocate(graph.n());
    set_max_degree(r_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(r_ctx.lp.max_num_neighbors);
  }

  void initialize(const Graph &graph) { _graph = &graph; }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    ASSERT(_graph == &p_graph.graph());
    ASSERT(p_graph.k() <= p_ctx.k);
    _p_graph = &p_graph;
    _p_ctx = &p_ctx;
    Base::initialize(_graph, _p_ctx->k);

    const std::size_t max_iterations = _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Label Propagation");
      if (perform_iteration() == 0) {
        return false;
      }
    }

    return true;
  }

  using Base::expected_total_gain;

public:
  [[nodiscard]] BlockID initial_cluster(const NodeID u) { return _p_graph->block(u); }

  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) { return _p_graph->block_weight(b); }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID b) { return _p_graph->block_weight(b); }

  bool move_cluster_weight(const BlockID old_block, const BlockID new_block, const BlockWeight delta,
                           const BlockWeight max_weight) {
    return _p_graph->try_move_block_weight(old_block, new_block, delta, max_weight);
  }

  void init_cluster(const NodeID /* u */, const BlockID /* b */) {}

  void init_cluster_weight(const BlockID /* b */, const BlockWeight /* weight */) {}

  [[nodiscard]] BlockID cluster(const NodeID u) { return _p_graph->block(u); }
  void move_node(const NodeID u, const BlockID block) { _p_graph->set_block<false>(u, block); }
  [[nodiscard]] BlockID num_clusters() { return _p_graph->k(); }
  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID block) { return _p_ctx->block_weights.max(block); }

  bool accept_cluster(const Base::ClusterSelectionState &state) {
    static_assert(std::is_signed_v<NodeWeight>);

    const NodeWeight current_max_weight = max_cluster_weight(state.current_cluster);
    const NodeWeight best_overload = state.best_cluster_weight - max_cluster_weight(state.best_cluster);
    const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
    const NodeWeight initial_overload = state.initial_cluster_weight - max_cluster_weight(state.initial_cluster);

    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain &&
             (current_overload < best_overload ||
              (current_overload == best_overload && state.local_rand.random_bool())))) &&
           (state.current_cluster_weight + state.u_weight < current_max_weight || current_overload < initial_overload ||
            state.current_cluster == state.initial_cluster);
  }

  const Graph *_graph{nullptr};
  PartitionedGraph *_p_graph{nullptr};
  const PartitionContext *_p_ctx;
  const RefinementContext &_r_ctx;
};

//
// Exposed wrapper
//

LabelPropagationRefiner::LabelPropagationRefiner(const Graph &graph, const RefinementContext &r_ctx)
    : _impl{std::make_unique<LabelPropagationRefinerImpl>(graph, r_ctx)} {}

LabelPropagationRefiner::~LabelPropagationRefiner() = default;

void LabelPropagationRefiner::initialize(const Graph &graph) { _impl->initialize(graph); }

bool LabelPropagationRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return _impl->refine(p_graph, p_ctx);
}

EdgeWeight LabelPropagationRefiner::expected_total_gain() const { return _impl->expected_total_gain(); }
} // namespace kaminpar