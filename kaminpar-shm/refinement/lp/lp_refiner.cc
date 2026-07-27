/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

#include <limits>
#include <span>

#include "kaminpar-shm/algorithms/iteration_order.h"
#include "kaminpar-shm/algorithms/label_propagation/balanced_node_processor.h"
#include "kaminpar-shm/algorithms/label_propagation/balanced_workspace.h"
#include "kaminpar-shm/algorithms/label_propagation/node_processing_kernel.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

template <typename Graph> class LPRefinementRunner {
  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  LPRefinementRunner(
      const Graph &graph,
      PartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      const LabelPropagationRefinementContext &lp_ctx,
      const std::span<const NodeID> communities,
      lp::BalancedWorkspace &workspace,
      InOrderIterationOrder &in_order,
      ChunkShuffledIterationOrder &chunk_shuffled
  )
      : _graph(graph),
        _p_graph(p_graph),
        _p_ctx(p_ctx),
        _lp_ctx(lp_ctx),
        _communities(communities),
        _workspace(workspace),
        _in_order(in_order),
        _chunk_shuffled(chunk_shuffled) {}

  bool run() {
    _workspace.ensure_capacity(_p_ctx.k);
    _workspace.state.reset(_p_graph, _p_ctx, _communities);
    initialize_iteration_order();

    lp::BalancedNodeProcessor processor(
        _graph, _workspace.state, _workspace.rating_maps, _workspace.statistics, _p_ctx.k
    );

    const std::size_t max_iterations =
        _lp_ctx.num_iterations == 0 ? kInfiniteIterations : _lp_ctx.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER(iteration == 0 ? "Initial iteration" : "Remaining iterations");
      _workspace.begin_round();

      auto kernel = lp::make_single_phase_kernel(processor);
      for_each(kernel);
      if (_workspace.statistics.totals().moved == 0) {
        break;
      }
    }

    return true;
  }

private:
  void initialize_iteration_order() {
    if (_lp_ctx.iteration_order == LabelPropagationIterationOrder::IN_ORDER) {
      _in_order.initialize(_graph.n());
    } else {
      _chunk_shuffled.initialize(_graph);
    }
  }

  template <typename Kernel> void for_each(Kernel &kernel) {
    if (_lp_ctx.iteration_order == LabelPropagationIterationOrder::IN_ORDER) {
      _in_order.for_each(kernel);
    } else {
      _chunk_shuffled.for_each(kernel);
    }
  }

  const Graph &_graph;
  PartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;
  const LabelPropagationRefinementContext &_lp_ctx;
  std::span<const NodeID> _communities;
  lp::BalancedWorkspace &_workspace;
  InOrderIterationOrder &_in_order;
  ChunkShuffledIterationOrder &_chunk_shuffled;
};

} // namespace

class LPRefinerImplWrapper {
public:
  explicit LPRefinerImplWrapper(const Context &ctx)
      : _lp_ctx(ctx.refinement.lp),
        _chunk_shuffled(_permutations) {}

  void initialize(const PartitionedGraph &) {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    SCOPED_HEAP_PROFILER("Label Propagation");
    SCOPED_TIMER("Label Propagation");

    return reified(p_graph, [&]<typename ConcreteGraph>(const ConcreteGraph &graph) {
      return LPRefinementRunner<ConcreteGraph>(
                 graph,
                 p_graph,
                 p_ctx,
                 _lp_ctx,
                 _communities,
                 _workspace,
                 _in_order,
                 _chunk_shuffled
      )
          .run();
    });
  }

  void set_communities(const std::span<const NodeID> communities) {
    _communities = communities;
  }

private:
  const LabelPropagationRefinementContext &_lp_ctx;
  lp::BalancedWorkspace _workspace;

  ChunkShuffledIterationOrder::Permutations _permutations;
  InOrderIterationOrder _in_order;
  ChunkShuffledIterationOrder _chunk_shuffled;

  std::span<const NodeID> _communities;
};

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
