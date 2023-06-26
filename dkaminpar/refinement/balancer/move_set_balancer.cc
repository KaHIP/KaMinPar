#include "dkaminpar/refinement/balancer/move_set_balancer.h"

namespace kaminpar::dist {
MoveSetBalancerMemoryContext::MoveSetBalancerMemoryContext(class MoveSetBalancerFactory *factory)
    : _factory(factory) {}

void MoveSetBalancerMemoryContext::free() {
  _factory->reclaim_m_ctx(std::move(*this));
}

MoveSetBalancerFactory::MoveSetBalancerFactory(const Context &ctx) : _ctx(ctx), _m_ctx(this) {}

std::unique_ptr<GlobalRefiner> MoveSetBalancerFactory::create(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  return std::make_unique<MoveSetBalancer>(_ctx, p_graph, p_ctx, std::move(_m_ctx));
}

void MoveSetBalancerFactory::reclaim_m_ctx(MoveSetBalancerMemoryContext m_ctx) {
  _m_ctx = std::move(m_ctx);
}

MoveSetBalancer::MoveSetBalancer(
    const Context &ctx,
    DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    MoveSetBalancerMemoryContext m_ctx
)
    : _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _m_ctx(std::move(m_ctx)) {}

MoveSetBalancer::~MoveSetBalancer() {
  _m_ctx.free();
}

void MoveSetBalancer::initialize() {}

bool MoveSetBalancer::refine() {
  return false;
}
} // namespace kaminpar::dist
