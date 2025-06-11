/*******************************************************************************
 * MultiQueue-based balancer for greedy minimum block weight balancing.
 *
 * @file:   underload_balancer.cc
 * @author: Daniel Seemaier
 * @date:   11.06.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/balancer/underload_balancer.h"

#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

UnderloadBalancer::UnderloadBalancer(const Context &ctx) : _ctx(ctx) {}

UnderloadBalancer::~UnderloadBalancer() = default;

std::string UnderloadBalancer::name() const {
  return "Underload Balancer";
}

void UnderloadBalancer::initialize(const PartitionedGraph & /* p_graph */) {}

bool UnderloadBalancer::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Underload Balancer");
  SCOPED_HEAP_PROFILER("Underload Balancer");

  // Terminate immediately if there is nothing to do
  if (!p_ctx.has_min_block_weights() || metrics::is_min_balanced(p_graph, p_ctx)) {
    return false;
  }

  return false;
}

} // namespace kaminpar::shm
