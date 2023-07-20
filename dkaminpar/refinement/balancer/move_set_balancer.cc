/*******************************************************************************
 * Greedy balancing algorithm that moves sets of nodes at a time.
 *
 * @file:   move_set_balancer.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "dkaminpar/refinement/balancer/move_set_balancer.h"

#include "dkaminpar/mpi/sparse_alltoall.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/refinement/balancer/move_sets.h"

namespace kaminpar::dist {
SET_DEBUG(true);

namespace {
template <typename Buffer, typename Combiner>
Buffer perform_binary_reduction(Buffer sendbuf, Combiner &&combiner, MPI_Comm comm) {
  enum class Role {
    SENDER,
    RECEIVER,
    NOOP
  };

  const PEID rank = mpi::get_comm_rank(comm);
  const PEID size = mpi::get_comm_size(comm);
  PEID active = size;

  while (active > 1) {
    if (rank >= active) {
      continue;
    }

    const Role role = [&] {
      if (rank == 0 && active % 2 == 1) {
        return Role::NOOP;
      } else if (rank < std::ceil(active / 2.0)) {
        return Role::RECEIVER;
      } else {
        return Role::SENDER;
      }
    }();

    if (role == Role::SENDER) {
      const PEID to = rank - active / 2;
      mpi::send(sendbuf.data(), sendbuf.size(), to, 0, comm);
      return {};
    } else if (role == Role::RECEIVER) {
      const PEID from = rank + active / 2;
      Buffer recvbuf = mpi::probe_recv<typename Buffer::value_type, Buffer>(from, 0, comm);
      sendbuf = combiner(std::move(sendbuf), std::move(recvbuf));
    }

    active = active / 2 + active % 2;
  }

  return sendbuf;
}
} // namespace

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

  for (const NodeID set : _move_sets.sets()) {
    if (!is_overloaded(_move_sets.block(set))) {
      continue;
    }
    try_pq_insertion(set);
  }
}

void MoveSetBalancer::try_pq_insertion(const NodeID set) {
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
    accept = relative_gain > min_key || (relative_gain == min_key && _rand.random_bool());
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
  std::vector<BlockWeight> tmp_block_weight_deltas(_p_graph.k());

  for (const auto &candidate : candidates) { // empty on non-ROOT
    if (candidate.from == candidate.to) {
      continue;
    }
    tmp_block_weight_deltas[candidate.from] -= candidate.weight;
    tmp_block_weight_deltas[candidate.to] += candidate.weight;
  }
  BlockID to = 0;
  for (auto &candidate : candidates) {
    if (candidate.from != candidate.to) {
      continue;
    }

    while (candidate.from == to || overload(to) < candidate.weight + tmp_block_weight_deltas[to]) {
      ++to;
      if (to >= _p_ctx.k) {
        to = 0;
      }
    }

    candidate.to = to;
    tmp_block_weight_deltas[candidate.from] -= candidate.weight;
    tmp_block_weight_deltas[candidate.to] += candidate.weight;
  }

  // Step 3: broadcast winners
  const std::size_t num_candidates = mpi::bcast(candidates.size(), 0, _p_graph.communicator());
  candidates.resize(num_candidates);
  mpi::bcast(candidates.data(), num_candidates, 0, _p_graph.communicator());

  // Step 4: apply changes
  perform_moves(candidates);
}

void MoveSetBalancer::perform_moves(const std::vector<MoveCandidate> &candidates) {
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());
  const PEID size = mpi::get_comm_size(_p_graph.communicator());

  struct MoveMessage {
    NodeID node;
    BlockID to;
  };
  Marker<> created_message_for_pe(size);
  std::vector<std::vector<MoveMessage>> move_sendbufs(size);

  for (const auto &candidate : candidates) {
    if (rank == candidate.owner) {
      _move_sets.move_set(candidate.set, candidate.from, candidate.to);
      for (NodeID u : _move_sets.elements(candidate.set)) {
        _p_graph.set_block<false>(u, candidate.to);

        for (const auto &[e, v] : _p_graph.neighbors(u)) {
          if (_p_graph.is_ghost_node(v)) {
            const PEID pe = _p_graph.ghost_owner(v);
            if (!created_message_for_pe.get(pe)) {
              move_sendbufs[pe].emplace_back(u, candidate.to);
              created_message_for_pe.set(pe);
            }
            continue;
          }

          if (!is_overloaded(_p_graph.block(v))) {
            continue;
          }

          const NodeID set = _move_sets.set_of(v);
          if (!_pqs.contains(set)) {
            // @todo update _pqs
            try_pq_insertion(set);
          }
        }
      }
    }

    // Update block weights
    _p_graph.set_block_weight(
        candidate.from, _p_graph.block_weight(candidate.from) - candidate.weight
    );
    _p_graph.set_block_weight(candidate.to, _p_graph.block_weight(candidate.to) + candidate.weight);
  }

  mpi::sparse_alltoall<MoveMessage>(
      move_sendbufs,
      [&](const auto recvbuf, const PEID pe) {
        for (const auto &[their_lnode, to] : recvbuf) {
          const GlobalNodeID gnode = their_lnode + _p_graph.offset_n(pe);
          const NodeID lnode = _p_graph.global_to_local_node(gnode);
          _move_sets.move_ghost_node(lnode, _p_graph.block(lnode), to);
          _p_graph.set_block<false>(lnode, to);
        }
      },
      _p_graph.communicator()
  );
}

std::vector<MoveSetBalancer::MoveCandidate> MoveSetBalancer::pick_sequential_candidates() {
  std::vector<MoveCandidate> candidates;

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

      candidates.push_back(MoveCandidate{
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

std::vector<MoveSetBalancer::MoveCandidate>
MoveSetBalancer::reduce_sequential_candidates(std::vector<MoveCandidate> candidates) {
  return perform_binary_reduction(
      candidates,
      [&](std::vector<MoveCandidate> lhs, std::vector<MoveCandidate> rhs) {
        // Precondition: candidates must be sorted by their from blocks
        auto check_sorted_by_from = [](const auto &candidates) {
          for (std::size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].from < candidates[i - 1].from) {
              return false;
            }
          }
          return true;
        };
        KASSERT(
            check_sorted_by_from(rhs) && check_sorted_by_from(lhs),
            "rhs or lhs candidates are not sorted by their .from property"
        );

        std::size_t idx_lhs = 0;
        std::size_t idx_rhs = 0;
        std::vector<BlockWeight> target_block_weight_deltas(_p_graph.k());

        while (idx_lhs < lhs.size() && idx_rhs < rhs.size()) {
          const BlockID from = std::min(lhs[idx_lhs].from, rhs[idx_rhs].from);

          // Find regions in `rhs` and `lhs` with move sets in block `from`
          std::size_t idx_lhs_end = idx_lhs;
          std::size_t idx_rhs_end = idx_rhs;
          while (idx_lhs_end < lhs.size() && lhs[idx_lhs_end].from == from) {
            ++idx_lhs_end;
          }
          while (idx_rhs_end < rhs.size() && rhs[idx_rhs_end].from == from) {
            ++idx_rhs_end;
          }

          // Merge regions
          const std::size_t lhs_count = idx_lhs_end - idx_lhs;
          const std::size_t rhs_count = idx_rhs_end - idx_rhs;
          const std::size_t count = lhs_count + rhs_count;

          std::vector<MoveCandidate> candidates(count);
          std::copy(lhs.begin() + idx_lhs, lhs.begin() + idx_lhs_end, candidates.begin());
          std::copy(
              rhs.begin() + idx_rhs, rhs.begin() + idx_rhs_end, candidates.begin() + lhs_count
          );
          std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            return a.gain > b.gain;
          });

          // Pick feasible prefix
          NodeWeight total_weight = 0;
          NodeID added_to_ans = 0;
        }

        // Keep remaining nodes
      },
      _p_graph.communicator()
  );
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
