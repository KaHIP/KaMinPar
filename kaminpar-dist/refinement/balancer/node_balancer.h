/*******************************************************************************
 * Distributed balancing algorithm that moves individual nodes.
 *
 * @file:   node_balancer.h
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 ******************************************************************************/
#pragma once

#include <variant>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {

class NodeBalancerFactory : public GlobalRefinerFactory {
  using CSRGainCache = OnTheFlyGainCache<DistributedCSRGraph>;
  using CompressedGainCache = OnTheFlyGainCache<DistributedCompressedGraph>;

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
  std::variant<std::monostate, CSRGainCache, CompressedGainCache> _gain_cache;
};

template <typename GainCache>
class NodeBalancerWithDecoupledGainCacheFactory : public GlobalRefinerFactory {
public:
  NodeBalancerWithDecoupledGainCacheFactory(const Context &ctx, GainCache &gain_cache);

  NodeBalancerWithDecoupledGainCacheFactory(const NodeBalancerWithDecoupledGainCacheFactory &) =
      delete;
  NodeBalancerWithDecoupledGainCacheFactory &
  operator=(const NodeBalancerWithDecoupledGainCacheFactory &) = delete;

  NodeBalancerWithDecoupledGainCacheFactory(NodeBalancerWithDecoupledGainCacheFactory &&) noexcept =
      default;
  NodeBalancerWithDecoupledGainCacheFactory &
  operator=(NodeBalancerWithDecoupledGainCacheFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
  GainCache &_gain_cache;
};

} // namespace kaminpar::dist
