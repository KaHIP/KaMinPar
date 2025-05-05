/*******************************************************************************
 * Sequential greedy balancing algorithm.
 *
 * @file:   sequential_greedy_balancer.cc
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/balancer/sequential_greedy_balancer.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

SequentialGreedyBalancer::SequentialGreedyBalancer([[maybe_unused]] const Context &ctx)
    : _csr_impl(std::make_unique<SequentialGreedyBalancerCSRImpl>()),
      _compressed_impl(std::make_unique<SequentialGreedyBalancerCompressedImpl>()) {}

SequentialGreedyBalancer::~SequentialGreedyBalancer() = default;

[[nodiscard]] std::string SequentialGreedyBalancer::name() const {
  return "Sequential Greedy Balancer";
}

void SequentialGreedyBalancer::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool SequentialGreedyBalancer::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Greedy Balancer");
  SCOPED_HEAP_PROFILER("Greedy Balancer");

  return reified(
      p_graph,
      [&](const CSRGraph &graph) {
        auto result = _csr_impl->balance(p_graph, graph, p_ctx.max_block_weights());
        return result.rebalanced;
      },
      [&](const CompressedGraph &graph) {
        auto result = _compressed_impl->balance(p_graph, graph, p_ctx.max_block_weights());
        return result.rebalanced;
      }
  );
}

} // namespace kaminpar::shm
