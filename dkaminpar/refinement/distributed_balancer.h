/*******************************************************************************
 * @file:   distributed_balancer.h
 *
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar {
class DistributedBalancer {
public:
    DistributedBalancer(const Context& ctx);

    DistributedBalancer(const DistributedBalancer&) = delete;
    DistributedBalancer& operator=(const DistributedBalancer&) = delete;

    DistributedBalancer(DistributedBalancer&&) noexcept = default;
    DistributedBalancer& operator=(DistributedBalancer&&) noexcept = default;

    void initialize(const DistributedPartitionedGraph& p_graph);
    void balance(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx);
};
}; // namespace dkaminpar

