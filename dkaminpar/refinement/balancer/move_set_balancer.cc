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
      _pqs(_p_graph.n(), _p_graph.k()),
      _pq_weights(_p_graph.k()),
      _moved_marker(_p_graph.n()),
      _move_sets(build_greedy_move_sets(_p_graph, _p_ctx, 5, std::move(m_ctx.move_sets_m_ctx))) {}

MoveSetBalancer::~MoveSetBalancer() {
  _factory.take_m_ctx(std::move(*this));
}

MoveSetBalancer::operator MoveSetBalancerMemoryContext() && {
  return {
      std::move(_move_sets),
  };
}

void MoveSetBalancer::initialize() {
  Random &rand = Random::instance();

  for (const NodeID set : _move_sets.sets()) {
    if (!is_overloaded(_move_sets.block(set))) {
      continue;
    }

    const BlockID from_block = _move_sets.block(set);
    const auto [relative_gain, to_block] = _move_sets.find_max_relative_gain(set);

    // Add weight to the correct weight bucket
    _weight_buckets.add(from_block, relative_gain);

    // Add this move set to the PQ if:
    // - we do not have enough move sets yet to remove all excess weight from the block
    bool accept = _pq_weights[from_block] < overload(from_block);
    bool replace_min = false;
    // - or its relative gain is better than the worst relative gain in the PQ
    if (!accept) {
      const double min_key = _pqs.peek_min_key(from_block);
      accept = relative_gain > min_key || (relative_gain == min_key && rand.random_bool());
      replace_min = true; // no effect if accept == false
    }

    if (accept) {
      if (replace_min) {
        NodeID replaced_set = _pqs.peek_min_id(from_block);
        _pqs.pop_min(from_block);
        _pq_weights[from_block] -= _move_sets.weight(replaced_set);
        _pq_weights[to_block] += _move_sets.weight(set);
      }

      _pqs.push(from_block, set, relative_gain);
    }
  }
}

bool MoveSetBalancer::refine() {
  return false;
}

BlockWeight MoveSetBalancer::overload(const BlockID block) const {
  static_assert(std::is_signed_v<BlockWeight>);
  return std::max<BlockWeight>(
      0, _p_ctx.graph->max_block_weight(block) - _p_graph.block_weight(block)
  );
}

bool MoveSetBalancer::is_overloaded(const BlockID block) const {
  return overload(block) > 0;
}
} // namespace kaminpar::dist
