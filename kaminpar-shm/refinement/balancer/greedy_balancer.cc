/*******************************************************************************
 * Greedy balancing algorithms that uses one thread per overloaded block.
 *
 * @file:   greedy_balancer.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/balancer/greedy_balancer.h"

namespace kaminpar::shm {

GreedyBalancer::GreedyBalancer(const Context &ctx)
    : _csr_impl(std::make_unique<GreedyBalancerCSRImpl>(ctx)),
      _compressed_impl(std::make_unique<GreedyBalancerCompressedImpl>(ctx)) {
  _memory_context.rating_map = tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>>{[&] {
    return RatingMap<EdgeWeight, NodeID>{ctx.partition.k};
  }};
}

GreedyBalancer::~GreedyBalancer() = default;

std::string GreedyBalancer::name() const {
  return "Greedy Balancer";
}

void GreedyBalancer::initialize(const PartitionedGraph &) {}

bool GreedyBalancer::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Greedy Balancer");
  SCOPED_HEAP_PROFILER("Greedy Balancer");

  const NodeWeight initial_overload = metrics::total_overload(p_graph, p_ctx);
  if (initial_overload == 0) {
    return false;
  }

  const auto balance = [&](auto &impl) {
    impl.setup(std::move(_memory_context));
    const bool found_improvement = impl.refine(p_graph, p_ctx);
    _memory_context = impl.release();
    return found_improvement;
  };

  if (p_graph.graph().is_compressed()) {
    return balance(*_compressed_impl);
  } else {
    return balance(*_csr_impl);
  }
}

} // namespace kaminpar::shm
