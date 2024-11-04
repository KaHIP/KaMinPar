/*******************************************************************************
 * Distributed balancing algorithm that moves individual nodes.
 *
 * @file:   node_balancer.h
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {

class NodeBalancerFactory : public GlobalRefinerFactory {
public:
  NodeBalancerFactory(const Context &ctx);

  NodeBalancerFactory(const NodeBalancerFactory &) = delete;
  NodeBalancerFactory &operator=(const NodeBalancerFactory &) = delete;

  NodeBalancerFactory(NodeBalancerFactory &&) noexcept = default;
  NodeBalancerFactory &operator=(NodeBalancerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

} // namespace kaminpar::dist
