/*******************************************************************************
 * @file:   distributed_balancer.cc
 *
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#include "dkaminpar/refinement/distributed_balancer.h"

namespace dkaminpar {
DistributedBalancer::DistributedBalancer(const Context& ctx) {
    ((void)ctx);
}

void DistributedBalancer::initialize(const DistributedPartitionedGraph& p_graph) {
    ((void)p_graph);
}
void DistributedBalancer::balance(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    ((void)p_graph);
    ((void)p_ctx);
}
} // namespace dkaminpar
