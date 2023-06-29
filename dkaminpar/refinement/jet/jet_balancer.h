/*******************************************************************************
 * @file:   jet_balancer.h
 * @author: Daniel Seemaier
 * @date:   29.06.2023
 * @brief:  Distributed JET balancer due to:
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class JetBalancerFactory : public GlobalRefinerFactory {
public:
  JetBalancerFactory(const Context &ctx);

  JetBalancerFactory(const JetBalancerFactory &) = delete;
  JetBalancerFactory &operator=(const JetBalancerFactory &) = delete;

  JetBalancerFactory(JetBalancerFactory &&) noexcept = default;
  JetBalancerFactory &operator=(JetBalancerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class JetBalancer : public GlobalRefiner {
  SET_STATISTICS_FROM_GLOBAL();
  SET_DEBUG(true);

public:
  JetBalancer(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  JetBalancer(const JetBalancer &) = delete;
  JetBalancer &operator=(const JetBalancer &) = delete;

  JetBalancer(JetBalancer &&) noexcept = default;
  JetBalancer &operator=(JetBalancer &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  const Context &_ctx;

  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;
};
} // namespace kaminpar::dist
