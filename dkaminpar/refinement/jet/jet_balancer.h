/*******************************************************************************
 * Distributed JET balancer due to: "Jet: Multilevel Graph Partitioning on GPUs"
 * by Gilbert et al.
 *
 * @file:   jet_balancer.h
 * @author: Daniel Seemaier
 * @date:   29.06.2023
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/refinement/gain_calculator.h"
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

namespace jet {
class Buckets {
public:
  constexpr static std::size_t kNumBuckets = 16;

  Buckets(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

  Buckets(
      const DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      StaticArray<GlobalNodeWeight> compactified_sizes
  );

  void init(const GainCalculator &gain_calculator);
  void clear();

  GlobalNodeWeight &size(BlockID block, std::size_t bucket);
  GlobalNodeWeight size(BlockID block, std::size_t bucket) const;

  StaticArray<GlobalNodeWeight> compactify() const;

  static std::size_t compute_bucket(EdgeWeight gain);

private:
  const DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  StaticArray<GlobalNodeWeight> _bucket_sizes;
};
} // namespace jet

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
  void weak_iteration();
  void strong_iteration();

  StaticArray<GlobalNodeWeight>
  compute_compacitifed_global_bucket_sizes(StaticArray<GlobalNodeWeight> local_bucket_sizes);

  bool is_overloaded_block(BlockID block) const;
  BlockWeight block_overload(BlockID block) const;

  int num_weak_iterations() const;
  int num_strong_iterations() const;

  const Context &_ctx;
  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  GainCalculator _gain_calculator;

  jet::Buckets _local_buckets;
};
} // namespace kaminpar::dist
