/*******************************************************************************
 * Greedy balancing algorithm that moves sets of nodes at a time.
 *
 * @file:   move_set_balancer.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "dkaminpar/refinement/balancer/move_set_balancer.h"

#include <iomanip>
#include <sstream>

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
      _move_sets(build_greedy_move_sets(
          _ctx.refinement.move_set_balancer.move_set_strategy,
          _p_graph,
          _p_ctx,
          5,
          std::move(m_ctx.move_sets_m_ctx)
      )) {}

MoveSetBalancer::~MoveSetBalancer() {
  _factory.take_m_ctx(std::move(*this));
}

MoveSetBalancer::operator MoveSetBalancerMemoryContext() && {
  return {
      std::move(_move_sets),
  };
}

void MoveSetBalancer::initialize() {
  KASSERT(
      [&] {
        for (const NodeID node : _p_graph.nodes()) {
          if (is_overloaded(_p_graph.block(node)) && _move_sets.set_of(node) == kInvalidNodeID) {
            LOG_ERROR << "node " << node << " is in block " << _p_graph.block(node)
                      << " with weight " << _p_graph.block_weight(_p_graph.block(node)) << " > "
                      << _p_ctx.graph->max_block_weight(_p_graph.block(node))
                      << ", but the node is not contained in any move set";
            return false;
          }
        }
        return true;
      }(),
      "move sets do not cover all nodes in overloaded blocks",
      assert::heavy
  );

  for (const NodeID set : _move_sets.sets()) {
    if (!is_overloaded(_move_sets.block(set))) {
      continue;
    }
    try_pq_insertion(set);
  }
}

void MoveSetBalancer::rebuild_move_sets() {
  _move_sets = build_greedy_move_sets(
      _ctx.refinement.move_set_balancer.move_set_strategy,
      _p_graph,
      _p_ctx,
      5,
      std::move(_move_sets)
  );
  clear();
  initialize();
}

void MoveSetBalancer::clear() {
  _pqs.clear();
  std::fill(_pq_weights.begin(), _pq_weights.end(), 0);
  _moved_marker.reset();
  _weight_buckets.clear();
}

void MoveSetBalancer::try_pq_insertion(const NodeID set) {
  KASSERT(!_pqs.contains(set));

  const BlockID from_block = _move_sets.block(set);
  const auto [relative_gain, to_block] = _move_sets.find_max_relative_gain(set);

  // Add weight to the correct weight bucket
  _weight_buckets.add(from_block, relative_gain);

  // Add this move set to the PQ if:
  bool accept = _ctx.refinement.move_set_balancer.seq_full_pq;
  bool replace_min = false;

  // - we do not have enough move sets yet to remove all excess weight from the block
  if (!accept) {
    accept = _pq_weights[from_block] < overload(from_block);
  }

  // - or its relative gain is better than the worst relative gain in the PQ
  if (!accept) {
    KASSERT(!_pqs.empty(from_block));
    const double min_key = _pqs.peek_min_key(from_block);
    accept = relative_gain > min_key || (relative_gain == min_key && _rand.random_bool());
    replace_min = true; // no effect if accept == false
  }

  if (accept) {
    if (replace_min) {
      KASSERT(!_pqs.empty(from_block));
      NodeID replaced_set = _pqs.peek_min_id(from_block);
      _pqs.pop_min(from_block);
      _pq_weights[from_block] -= _move_sets.weight(replaced_set);
      _pq_weights[to_block] += _move_sets.weight(set);
    }

    _pqs.push(from_block, set, relative_gain);
  }
}

void MoveSetBalancer::try_pq_update(const NodeID set) {
  const BlockID from_block = _move_sets.block(set);
  const auto [relative_gain, to_block] = _move_sets.find_max_relative_gain(set);

  KASSERT(_pqs.contains(set), "set " << set << " not contained in the PQ");
  KASSERT(relative_gain != std::numeric_limits<double>::min(), "illegal relative gain");
  KASSERT(relative_gain != std::numeric_limits<double>::max(), "illegal relative gain");
  _pqs.change_priority(from_block, set, relative_gain);
}

bool MoveSetBalancer::refine() {
  KASSERT(
      graph::debug::validate_partition(_p_graph),
      "input partition for the move set balancer is in an inconsistent state",
      assert::heavy
  );
  DBG0 << dbg_get_partition_state_str();

  const double initial_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
  double prev_imbalance_distance = initial_imbalance_distance;

  for (int round = 0; round < _ctx.refinement.move_set_balancer.max_num_rounds; ++round) {
    DBG << "Starting round " << round;

    if (round > 0 && _ctx.refinement.move_set_balancer.move_set_rebuild_interval > 0 &&
        (round % _ctx.refinement.move_set_balancer.move_set_rebuild_interval) == 0) {
      DBG << "  --> rebuild move sets after every "
          << _ctx.refinement.move_set_balancer.move_set_rebuild_interval;

      rebuild_move_sets();
    }

    if (_ctx.refinement.move_set_balancer.enable_sequential_balancing) {
      perform_sequential_round();
      DBG << "  --> Round " << round << ": seq. balancing: " << prev_imbalance_distance << " --> "
          << metrics::imbalance_l1(_p_graph, _p_ctx);
    }

    if (_ctx.refinement.move_set_balancer.enable_parallel_balancing) {
      const double imbalance_distance_after_sequential_balancing =
          metrics::imbalance_l1(_p_graph, _p_ctx);
      if ((prev_imbalance_distance - imbalance_distance_after_sequential_balancing) /
              prev_imbalance_distance <
          _ctx.refinement.move_set_balancer.parallel_threshold) {
        perform_parallel_round();
        DBG << "  --> Round " << round
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

    KASSERT(
        graph::debug::validate_partition(_p_graph),
        "partition is in an inconsistent state after round " << round,
        assert::heavy
    );
  }

  return prev_imbalance_distance > 0;
}

void MoveSetBalancer::perform_parallel_round() {
  constexpr static bool kUseBinaryReductionTree = true;

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  auto buckets = [&] {
    if constexpr (kUseBinaryReductionTree) {
      return perform_binary_reduction(
          _weight_buckets.compactify(),
          [&](auto lhs, auto rhs) {
            for (std::size_t i = 0; i < lhs.size(); ++i) {
              lhs[i] += rhs[i];
            }
            return std::move(lhs);
          },
          _p_graph.communicator()
      );
    } else {
      auto buckets = _weight_buckets.compactify();
      if (rank == 0) {
        MPI_Reduce(
            MPI_IN_PLACE,
            buckets.data(),
            buckets.size(),
            mpi::type::get<GlobalNodeWeight>(),
            MPI_SUM,
            0,
            _p_graph.communicator()
        );
      } else {
        MPI_Reduce(
            buckets.data(),
            nullptr,
            buckets.size(),
            mpi::type::get<GlobalNodeWeight>(),
            MPI_SUM,
            0,
            _p_graph.communicator()
        );
      }
    }
  }();

  // Determine cut-off buckets and broadcast them to all PEs
  const BlockID num_overloaded_blocks = count_overloaded_blocks();
  std::vector<int> cut_off_buckets(num_overloaded_blocks);
  std::vector<BlockID> to_overloaded_map(_p_graph.k());
  BlockID current_overloaded_block = 0;

  for (const BlockID block : _p_graph.blocks()) {
    BlockWeight current_weight = _p_graph.block_weight(block);
    const BlockWeight max_weight = _p_ctx.graph->max_block_weight(block);
    if (current_weight > max_weight) {
      if (rank == 0) {
        int cut_off_bucket = 0;
        for (; cut_off_bucket < Buckets::kNumBuckets && current_weight > max_weight;
             ++cut_off_bucket) {
          KASSERT(
              current_overloaded_block * Buckets::kNumBuckets + cut_off_bucket < buckets.size()
          );
          current_weight -=
              buckets[current_overloaded_block * Buckets::kNumBuckets + cut_off_bucket];
        }

        KASSERT(current_overloaded_block < cut_off_buckets.size());
        cut_off_buckets[current_overloaded_block] = cut_off_bucket;
      }

      KASSERT(block < to_overloaded_map.size());

      to_overloaded_map[block] = current_overloaded_block;
      ++current_overloaded_block;
    }
  }

  MPI_Bcast(cut_off_buckets.data(), num_overloaded_blocks, MPI_INT, 0, _p_graph.communicator());

  std::vector<MoveCandidate> candidates;
  std::vector<BlockWeight> block_weight_deltas(_p_graph.k());

  for (const NodeID set : _move_sets.sets()) {
    const BlockID from = _move_sets.block(set);

    if (is_overloaded(from)) {
      auto [gain, to] = _move_sets.find_max_relative_gain(set);
      const auto bucket = Buckets::compute_bucket(gain);

      if (bucket < cut_off_buckets[to_overloaded_map[from]]) {
        const NodeWeight weight = _move_sets.weight(set);

        MoveCandidate candidate = {
            .owner = rank,
            .set = set,
            .weight = weight,
            .gain = gain,
            .from = from,
            .to = to,
        };

        if (to == from) {
          [[maybe_unused]] const bool reassigned =
              assign_feasible_target_block(candidate, block_weight_deltas);
          KASSERT(reassigned);
        }

        block_weight_deltas[to] += weight;
        candidates.push_back(candidate);
      }
    }
  }

  MPI_Allreduce(
      MPI_IN_PLACE,
      block_weight_deltas.data(),
      asserting_cast<int>(block_weight_deltas.size()),
      mpi::type::get<BlockWeight>(),
      MPI_SUM,
      _p_graph.communicator()
  );

  Random &rand = Random::instance();
  std::size_t num_rejected_candidates = 0;
  std::vector<BlockWeight> actual_block_weight_deltas(_p_graph.k());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto &candidate = candidates[i];
    const double probability = 1.0 * underload(candidate.to) / block_weight_deltas[candidate.to];
    if (rand.random_bool(probability)) {
      actual_block_weight_deltas[candidate.to] += candidate.weight;
      actual_block_weight_deltas[candidate.from] -= candidate.weight;
    } else {
      ++num_rejected_candidates;
      std::swap(candidates[i], candidates[candidates.size() - num_rejected_candidates]);
      block_weight_deltas[candidate.to] -= candidate.weight;
    }
  }

  MPI_Allreduce(
      MPI_IN_PLACE,
      actual_block_weight_deltas.data(),
      asserting_cast<int>(actual_block_weight_deltas.size()),
      mpi::type::get<BlockWeight>(),
      MPI_SUM,
      _p_graph.communicator()
  );
  for (const BlockID block : _p_graph.blocks()) {
    _p_graph.set_block_weight(
        block, _p_graph.block_weight(block) + actual_block_weight_deltas[block]
    );
  }

  candidates.resize(candidates.size() - num_rejected_candidates);
  perform_moves(candidates, false);
}

void MoveSetBalancer::perform_sequential_round() {
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  // Step 1: identify the best move set candidates globally
  auto candidates = pick_sequential_candidates();
  DBG << "Picked " << candidates.size() << " candidates";

  candidates = reduce_sequential_candidates(pick_sequential_candidates());
  DBGC(rank == 0) << "Reduced to " << candidates.size() << " candidates";

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

    [[maybe_unused]] const bool reassigned =
        assign_feasible_target_block(candidate, tmp_block_weight_deltas);
    KASSERT(reassigned);

    tmp_block_weight_deltas[candidate.from] -= candidate.weight;
    tmp_block_weight_deltas[candidate.to] += candidate.weight;
  }

  // Step 3: broadcast winners
  const std::size_t num_candidates = mpi::bcast(candidates.size(), 0, _p_graph.communicator());
  candidates.resize(num_candidates);
  mpi::bcast(candidates.data(), num_candidates, 0, _p_graph.communicator());

  // Step 4: apply changes
  perform_moves(candidates, true);
}

void MoveSetBalancer::perform_moves(
    const std::vector<MoveCandidate> &candidates, const bool update_block_weights
) {
  DBG << "Perform " << candidates.size() << " moves, update block weights=" << update_block_weights;

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

      // We track moved sets to exclude them from further rounds
      _moved_marker.set(candidate.set);

      // If we perform a parallel round, the set might still be in the PQ --> remove it
      if (_pqs.contains(candidate.set)) {
        _pq_weights[candidate.from] -= candidate.weight;
        _pqs.remove(candidate.from, candidate.set);
      }

      for (NodeID u : _move_sets.elements(candidate.set)) {
        _p_graph.set_block<false>(u, candidate.to);

        for (const auto &[e, v] : _p_graph.neighbors(u)) {
          if (_p_graph.is_ghost_node(v)) {
            const PEID pe = _p_graph.ghost_owner(v);
            if (!created_message_for_pe.get(pe)) {
              move_sendbufs[pe].push_back({
                  .node = u,
                  .to = candidate.to,
              });
              created_message_for_pe.set(pe);
            }
            continue;
          }

          // !is_overloaded(.) is not a sufficient condition, since parallel moves might overload
          // new blocks that have not been overloaded when the move sets where created
          // --> also ignore sets that are not assigned to any move sets

          if (const NodeID set = _move_sets.set_of(v);
              is_overloaded(_p_graph.block(v)) && _move_sets.contains(v) && set != candidate.set &&
              !_moved_marker.get(set)) {
            if (!_pqs.contains(set)) {
              try_pq_insertion(set);
            } else {
              try_pq_update(set);
            }
          }
        }

        created_message_for_pe.reset();
      }
    }

    // Update block weights
    if (update_block_weights) {
      _p_graph.set_block_weight(
          candidate.from, _p_graph.block_weight(candidate.from) - candidate.weight
      );
      _p_graph.set_block_weight(
          candidate.to, _p_graph.block_weight(candidate.to) + candidate.weight
      );
    }
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
  DBG0 << dbg_get_pq_state_str();

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
        std::vector<BlockWeight> block_weight_deltas(_p_graph.k());

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
          std::size_t num_rejected_candidates = 0;
          std::size_t num_accepted_candidates = 0;

          for (std::size_t i = 0; i < count; ++i) {
            const BlockID to = candidates[i].to;
            const NodeWeight weight = candidates[i].weight;

            // Reject the move set candidate if it would overload the target block
            if (from != to && _p_graph.block_weight(to) + block_weight_deltas[to] + weight >
                                  _p_ctx.graph->max_block_weight(to)) {
              candidates[i].set = kInvalidNodeID;
              ++num_rejected_candidates;
            } else {
              block_weight_deltas[to] += weight;
              total_weight += weight;
              ++num_accepted_candidates;

              if (total_weight >= overload(from) ||
                  num_accepted_candidates >=
                      _ctx.refinement.move_set_balancer.seq_num_nodes_per_block) {
                break;
              }
            }
          }

          // Remove rejected candidates
          for (std::size_t i = 0; i < num_accepted_candidates; ++i) {
            while (candidates[i].set == kInvalidNodeID) {
              std::swap(
                  candidates[i], candidates[num_accepted_candidates + num_rejected_candidates - 1]
              );
              --num_rejected_candidates;
            }
          }
          candidates.resize(num_accepted_candidates);
        }

        // Keep remaining nodes
        auto add_remaining_candidates = [&](const auto &vec, std::size_t i) {
          for (; i < vec.size(); ++i) {
            const BlockID from = vec[i].from;
            const BlockID to = vec[i].to;
            const NodeWeight weight = vec[i].weight;

            if (from == to && _p_graph.block_weight(to) + block_weight_deltas[to] + weight <=
                                  _p_ctx.graph->max_block_weight(to)) {
              candidates.push_back(vec[i]);
              if (from != to) {
                block_weight_deltas[to] += weight;
              }
            }
          }
        };
        add_remaining_candidates(lhs, idx_lhs);
        add_remaining_candidates(rhs, idx_rhs);

        return candidates;
      },
      _p_graph.communicator()
  );
}

BlockWeight MoveSetBalancer::overload(const BlockID block) const {
  static_assert(std::is_signed_v<BlockWeight>);
  return std::max<BlockWeight>(
      0, _p_graph.block_weight(block) - _p_ctx.graph->max_block_weight(block)
  );
}

BlockWeight MoveSetBalancer::underload(const BlockID block) const {
  static_assert(std::is_signed_v<BlockWeight>);
  return std::max<BlockWeight>(
      0, _p_ctx.graph->max_block_weight(block) - _p_graph.block_weight(block)
  );
}

bool MoveSetBalancer::is_overloaded(const BlockID block) const {
  return overload(block) > 0;
}

BlockID MoveSetBalancer::count_overloaded_blocks() const {
  return metrics::num_imbalanced_blocks(_p_graph, _p_ctx);
}

bool MoveSetBalancer::assign_feasible_target_block(
    MoveCandidate &candidate, const std::vector<BlockWeight> &deltas
) const {
  do {
    ++candidate.to;
    if (candidate.to >= _p_ctx.k) {
      candidate.to = 0;
    }
  } while (candidate.from != candidate.to &&
           underload(candidate.to) < candidate.weight + deltas[candidate.to]);

  return candidate.from != candidate.to;
}

std::string MoveSetBalancer::dbg_get_partition_state_str() const {
  std::stringstream ss;
  ss << "Overloaded blocks: ";
  for (const BlockID block : _p_graph.blocks()) {
    if (is_overloaded(block)) {
      ss << "[" << std::setw(3) << block << ":" << std::setw(5) << overload(block) << "] ";
    }
  }
  return ss.str();
}

std::string MoveSetBalancer::dbg_get_pq_state_str() const {
  std::stringstream ss;
  ss << "PQ size: " << _pqs.size() << " -> ";
  for (const BlockID block : _p_graph.blocks()) {
    ss << "[" << std::setw(3) << block << ":" << std::setw(5) << _pqs.size(block) << "] ";
  }
  return ss.str();
}
} // namespace kaminpar::dist
