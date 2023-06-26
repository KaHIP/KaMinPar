#pragma once

#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class MoveSetBalancerMemoryContext {
public:
  MoveSetBalancerMemoryContext(class MoveSetBalancerFactory *factory);

  MoveSetBalancerMemoryContext(const MoveSetBalancerMemoryContext &) = delete;
  MoveSetBalancerMemoryContext &operator=(const MoveSetBalancerMemoryContext &) = delete;

  MoveSetBalancerMemoryContext(MoveSetBalancerMemoryContext &&) = default;
  MoveSetBalancerMemoryContext &operator=(MoveSetBalancerMemoryContext &&) = default;

  void free();

private:
  class MoveSetBalancerFactory *_factory = nullptr;
};

class MoveSetBalancerFactory : public GlobalRefinerFactory {
  friend MoveSetBalancerMemoryContext;

public:
  MoveSetBalancerFactory(const Context &ctx);

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  void reclaim_m_ctx(MoveSetBalancerMemoryContext m_ctx);

  const Context &_ctx;

  MoveSetBalancerMemoryContext _m_ctx;
};

class MoveSetBalancer : public GlobalRefiner {
public:
  MoveSetBalancer(
      const Context &ctx,
      DistributedPartitionedGraph &p_graph,
      const PartitionContext &p_ctx,
      MoveSetBalancerMemoryContext m_ctx
  );

  MoveSetBalancer(const MoveSetBalancer &) = delete;
  MoveSetBalancer &operator=(const MoveSetBalancer &) = delete;

  MoveSetBalancer(MoveSetBalancer &&) = delete;
  MoveSetBalancer &operator=(MoveSetBalancer &&) = delete;

  ~MoveSetBalancer();

  void initialize() final;
  bool refine() final;

private:
  const Context &_ctx;
  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;

  MoveSetBalancerMemoryContext _m_ctx;
};
} // namespace kaminpar::dist
