/*******************************************************************************
 * Greedy balancing algorithm that moves sets of nodes at a time.
 *
 * @file:   move_set_balancer.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "dkaminpar/refinement/balancer/move_set_balancer.h"

#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/refinement/balancer/move_sets.h"

namespace kaminpar::dist {
SET_DEBUG(true);

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
      _weight_buckets(_p_graph, _p_ctx),
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
  const double initial_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
  double prev_imbalance_distance = initial_imbalance_distance;

  for (int round = 0; round < _ctx.refinement.move_set_balancer.max_num_rounds; ++round) {
    if (_ctx.refinement.move_set_balancer.enable_sequential_balancing) {
      perform_sequential_round();
      DBG << "Round " << round << ": seq. balancing: " << prev_imbalance_distance << " --> "
          << metrics::imbalance_l1(_p_graph, _p_ctx);
    }

    if (_ctx.refinement.move_set_balancer.enable_parallel_balancing) {
      const double imbalance_distance_after_sequential_balancing =
          metrics::imbalance_l1(_p_graph, _p_ctx);
      if ((prev_imbalance_distance - imbalance_distance_after_sequential_balancing) /
              prev_imbalance_distance <
          _ctx.refinement.move_set_balancer.parallel_threshold) {
        perform_parallel_round();
        DBG << "Round " << round
            << ": par. balancing: " << imbalance_distance_after_sequential_balancing << " --> "
            << metrics::imbalance_l1(_p_graph, _p_ctx);
      }
    }

    // Abort if we couldn't improve balance
    const double next_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
    if (next_imbalance_distance >= prev_imbalance_distance) {
      DBG << "Stallmate: imbalance distance " << next_imbalance_distance
          << " could not be improved in round " << round;
      break;
    } else {
      prev_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
    }
  }

  return prev_imbalance_distance > 0;
}

void MoveSetBalancer::perform_parallel_round() {}

void MoveSetBalancer::perform_sequential_round() {
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  // Step 1: identify the best move set candidates globally
  auto candidates = reduce_sequential_candidates(pick_sequential_candidates());

  // Step 2: let ROOT decide which candidates to pick
  for (const auto &candidate : candidates) { // empty on non-ROOT
    if (candidate.from == candidate.to) {
      continue;
    }
    perform_sequential_move(candidate);
  }
  BlockID to = 0;
  for (auto &candidate : candidates) {
    if (candidate.from != candidate.to) {
      continue;
    }

    while (candidate.from == to || overload(to) < candidate.weight) {
      ++to;
      if (to >= _p_ctx.k) {
        to = 0;
      }
    }

    candidate.to = to;
    perform_sequential_move(candidate);
  }

  // Step 3: broadcast winners
}

void MoveSetBalancer::perform_sequential_move(const SequentialMoveCandidate &candidate) {

}

std::vector<MoveSetBalancer::SequentialMoveCandidate>
MoveSetBalancer::pick_sequential_candidates() {
  std::vector<SequentialMoveCandidate> candidates;

  for (const BlockID from : _p_graph.blocks()) {
    if (!is_overloaded(from)) {
      continue;
    }

    const std::size_t start = candidates.size();

    for (NodeID num = 0; num < _ctx.refinement.move_set_balancer.seq_num_nodes_per_block; ++num) {
      if (_pqs.empty(from)) {
        break;
      }

      const NodeID set = _pqs.peek_max_id(from);
      const double relative_gain = _pqs.peek_max_key(from);
      const NodeWeight weight = _move_sets.weight(set);
      _pqs.pop_max(from);

      // @todo we can avoid this recalculation in the sequential case
      auto [actual_relative_gain, to] = _move_sets.find_max_relative_gain(set);
      KASSERT(actual_relative_gain == relative_gain);

      candidates.push_back(SequentialMoveCandidate{
          .set = set,
          .weight = weight,
          .gain = actual_relative_gain,
          .from = from,
          .to = to,
      });
    }

    for (auto candidate = candidates.begin() + start; candidate != candidates.end(); ++candidate) {
      _pqs.push(from, candidate->set, candidate->gain);
    }
  }

  return candidates;
}

std::vector<MoveSetBalancer::SequentialMoveCandidate>
MoveSetBalancer::reduce_sequential_candidates(std::vector<SequentialMoveCandidate> candidates) {
  return std::move(candidates);
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
