/*******************************************************************************
 * Greedy balancing algorithm that moves sets of nodes at a time.
 *
 * @file:   move_set_balancer.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "dkaminpar/refinement/balancer/move_set_balancer.h"

#include "dkaminpar/refinement/balancer/move_sets.h"

namespace kaminpar::dist {
struct MoveSetBalancerMemoryContext {
  MoveSetsMemoryContext move_sets_m_ctx;
};

MoveSetBalancerFactory::MoveSetBalancerFactory(const Context &ctx)
    : _ctx(ctx),
      _m_ctx(std::make_unique<MoveSetBalancerMemoryContext>()) {}

MoveSetBalancerFactory::~MoveSetBalancerFactory() = default;

std::unique_ptr<GlobalRefiner> MoveSetBalancerFactory::create(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  return std::make_unique<MoveSetBalancer>(*this, _ctx, p_graph, p_ctx, std::move(*_m_ctx));
}

void MoveSetBalancerFactory::take_m_ctx(MoveSetBalancerMemoryContext m_ctx) {
  *_m_ctx = std::move(m_ctx);
}

MoveSetBalancer::MoveSetBalancer(
    MoveSetBalancerFactory &factory,
    const Context &ctx,
    DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    MoveSetBalancerMemoryContext m_ctx
)
    : _factory(factory),
      _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _move_sets(build_greedy_move_sets(_p_graph, _p_ctx, 5, std::move(m_ctx.move_sets_m_ctx))) {}

MoveSetBalancer::~MoveSetBalancer() {
  _factory.take_m_ctx(std::move(*this));
}

MoveSetBalancer::operator MoveSetBalancerMemoryContext() && {
  return {
      std::move(_move_sets),
  };
}

void MoveSetBalancer::initialize() {}

bool MoveSetBalancer::refine() {
  return false;
}
} // namespace kaminpar::dist
