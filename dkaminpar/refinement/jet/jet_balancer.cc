/*******************************************************************************
 * Distributed JET balancer due to: "Jet: Multilevel Graph Partitioning on GPUs" 
 * by Gilbert et al.
 *
 * @file:   jet_balancer.cc
 * @author: Daniel Seemaier
 * @date:   29.06.2023
 ******************************************************************************/
#include "dkaminpar/refinement/jet/jet_balancer.h"

namespace kaminpar::dist {
JetBalancerFactory::JetBalancerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
JetBalancerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<JetBalancer>(_ctx, p_graph, p_ctx);
}

JetBalancer::JetBalancer(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx) {}

void JetBalancer::initialize() {}

bool JetBalancer::refine() {
  return false;
}
} // namespace kaminpar::dist
