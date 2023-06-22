/*******************************************************************************
 * @file:   greedy_balancer.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#include "dkaminpar/refinement/greedy_balancer.h"

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/graphutils/synchronization.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/math.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
GreedyBalancer::GreedyBalancer(const Context &ctx)
    : _ctx(ctx),
      _pq(ctx.partition.graph->n, ctx.partition.k),
      _pq_weight(ctx.partition.k),
      _marker(ctx.partition.graph->n) {}

void GreedyBalancer::initialize(const DistributedGraph &) {}

void GreedyBalancer::refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Balancer");

  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  // Only balance the partition if it is infeasible
  if (metrics::is_feasible(*_p_graph, *_p_ctx)) {
    return;
  }

  IFSTATS(reset_statistics());
  IFSTATS(_stats.initial_feasible = metrics::is_feasible(p_graph, p_ctx));
  IFSTATS(_stats.initial_cut = metrics::edge_cut(p_graph));
  IFSTATS(_stats.initial_imbalance = metrics::imbalance(p_graph));
  IFSTATS(_stats.initial_num_imbalanced_blocks = metrics::num_imbalanced_blocks(p_graph, p_ctx));

  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  DBG << "Init PQ";
  init_pq();

  double previous_imbalance_distance =
      fast_balancing_enabled() ? metrics::imbalance_l1(p_graph, p_ctx) : 0.0;

  for (int round = 0; round < _ctx.refinement.greedy_balancer.max_num_rounds; round++) {
    if (metrics::is_feasible(*_p_graph, *_p_ctx)) {
      break;
    }

    // If balancing takes a very long time, print statistics periodically
    IFSTATS(++_stats.num_reduction_rounds);
    if constexpr (kStatistics) {
      if (round == kPrintStatsEveryNRounds) {
        print_statistics();
        round = 0;
      }
    }

    if (strong_balancing_enabled()) {
      // Pick best move candidates for each block
      START_TIMER("Pick move candidates");
      auto candidates = pick_move_candidates();
      STOP_TIMER();

      START_TIMER("Reduce move candidates");
      candidates = reduce_buckets_or_move_candidates(std::move(candidates));
      STOP_TIMER();

      START_TIMER("Perform moves on root PE");
      if (rank == 0) {
        // Move nodes that already have a target block
        for (const auto &move : candidates) {
          if (move.from != move.to) {
            IFSTATS(++_stats.num_adjacent_moves);
            perform_move(move);
          } else {
            IFSTATS(++_stats.num_nonadjacent_moves);
          }
        }

        // Move nodes that do not have a target block
        BlockID cur = 0;
        for (auto &candidate : candidates) {
          auto &[node, from, to, weight, rel_gain] = candidate;

          if (from == to) {
            // Look for next block that can take node
            while (cur == from ||
                   _p_graph->block_weight(cur) + weight > _p_ctx->graph->max_block_weight(cur)) {
              ++cur;
              if (cur >= _p_ctx->k) {
                cur = 0;
              }
            }

            to = cur;
            perform_move(candidate);
          }
        }
      }
      STOP_TIMER();

      // Broadcast winners
      START_TIMER("Broadcast reduction result");
      const std::size_t num_winners = mpi::bcast(candidates.size(), 0, _p_graph->communicator());
      candidates.resize(num_winners);
      mpi::bcast(candidates.data(), num_winners, 0, _p_graph->communicator());
      STOP_TIMER();

      START_TIMER("Perform moves");
      if (rank != 0) {
        perform_moves(candidates);
      }
      STOP_TIMER();

      KASSERT(
          graph::debug::validate_partition(*_p_graph),
          "balancer produced invalid partition",
          assert::heavy
      );

      if (num_winners == 0) {
        DBG0 << "No winners in round " << round << " of strong balancing --> TERMINATE";
        break;
      }
    }

    if (fast_balancing_enabled()) {
      const double current_imbalance_distance = metrics::imbalance_l1(p_graph, p_ctx);
      DBG0 << "Strong rebalancing improved imbalance from " << previous_imbalance_distance << " to "
           << current_imbalance_distance << " by "
           << (previous_imbalance_distance - current_imbalance_distance) /
                  previous_imbalance_distance
           << " vs " << _ctx.refinement.greedy_balancer.fast_balancing_threshold;

      if ((previous_imbalance_distance - current_imbalance_distance) / previous_imbalance_distance <
              _ctx.refinement.greedy_balancer.fast_balancing_threshold ||
          !strong_balancing_enabled()) {
        mpi::barrier(_p_graph->communicator());

        SCOPED_TIMER("Fast rebalancing");

        // Since we re-init the PQs anyways, we can use the marker to keep track of moved nodes
        _marker.reset();

        START_TIMER("Init buckets");
        init_buckets(); // @todo only do once, then keep up to date?
        MPI_Barrier(_p_graph->communicator());
        STOP_TIMER();

        START_TIMER("Compactify buckets");
        auto compact_buckets = compactify_buckets();
        MPI_Barrier(_p_graph->communicator());
        STOP_TIMER();

        START_TIMER("Reduce buckets");
        compact_buckets = reduce_buckets_or_move_candidates(std::move(compact_buckets));
        MPI_Barrier(_p_graph->communicator());
        STOP_TIMER();

        // Determine cut-off buckets on root
        START_TIMER("Finding cut-off buckets");
        const BlockID num_overloaded_blocks = metrics::num_imbalanced_blocks(p_graph, p_ctx);
        std::vector<int> cut_off_buckets(num_overloaded_blocks);
        std::vector<BlockID> to_overloaded_map(p_graph.k());
        BlockID current_overloaded_block = 0;

        for (const BlockID block : p_graph.blocks()) {
          BlockWeight current_weight = p_graph.block_weight(block);
          const BlockWeight max_weight = p_ctx.graph->max_block_weight(block);
          if (current_weight > max_weight) {
            if (rank == 0) {
              int cut_off_bucket = 0;
              for (; cut_off_bucket < kBucketsPerBlock && current_weight > max_weight;
                   ++cut_off_bucket) {
                KASSERT(
                    current_overloaded_block * kBucketsPerBlock + cut_off_bucket <
                    compact_buckets.size()
                );
                current_weight -=
                    compact_buckets[current_overloaded_block * kBucketsPerBlock + cut_off_bucket];
              }

              KASSERT(current_overloaded_block < cut_off_buckets.size());
              cut_off_buckets[current_overloaded_block] = cut_off_bucket;
            }

            KASSERT(block < to_overloaded_map.size());

            to_overloaded_map[block] = current_overloaded_block;
            ++current_overloaded_block;
          }
        }

        // Broadcast to other PEs
        MPI_Bcast(
            cut_off_buckets.data(), num_overloaded_blocks, MPI_INT, 0, p_graph.communicator()
        );
        STOP_TIMER();

        // Find move candidates
        std::vector<BlockID> target_blocks;
        for (const BlockID b : _p_graph->blocks()) {
          if (_p_graph->block_weight(b) < _p_ctx->graph->max_block_weight(b)) {
            target_blocks.push_back(b);
          }
        }
        KASSERT(!target_blocks.empty());

        std::size_t next_target_block_index = rank % target_blocks.size();

        std::vector<std::pair<NodeID, BlockID>> move_candidates;
        std::vector<BlockWeight> weight_to_block(p_graph.k());

        START_TIMER("Find move candidates");
        for (const NodeID node : _p_graph->nodes()) {
          const BlockID from = _p_graph->block(node);
          const BlockWeight overload = block_overload(from);

          if (overload > 0) { // Node in overloaded block
            const auto [max_gainer, rel_gain] = compute_gain(node, from);
            const auto bucket = get_bucket(rel_gain);

            KASSERT(from < to_overloaded_map.size());
            KASSERT(to_overloaded_map[from] < cut_off_buckets.size());

            if (bucket < cut_off_buckets[to_overloaded_map[from]]) {
              if (max_gainer != from) { // Node must be moved to its max gainer
                KASSERT(max_gainer < weight_to_block.size());

                weight_to_block[max_gainer] += _p_graph->node_weight(node);
                move_candidates.emplace_back(node, max_gainer);
              } else { // Node can be moved to any block -> pick target blocks round-robin
                BlockID new_to = 0;
                do {
                  ++next_target_block_index;
                  if (next_target_block_index >= target_blocks.size()) {
                    next_target_block_index = 0;
                  }
                  KASSERT(next_target_block_index < target_blocks.size());
                  new_to = target_blocks[next_target_block_index];
                } while (_p_graph->block_weight(new_to) + _p_graph->node_weight(node) >
                         _p_ctx->graph->max_block_weight(new_to));

                KASSERT(new_to < weight_to_block.size());
                weight_to_block[new_to] += _p_graph->node_weight(node);
                move_candidates.emplace_back(node, new_to);
              }
            }
          }
        }
        MPI_Barrier(_p_graph->communicator());
        STOP_TIMER();

        // Compute total weight to each block
        START_TIMER("Allreduce weight to block");
        MPI_Allreduce(
            MPI_IN_PLACE,
            weight_to_block.data(),
            p_graph.k(),
            mpi::type::get<BlockWeight>(),
            MPI_SUM,
            p_graph.communicator()
        );
        STOP_TIMER();

        // Perform moves
        START_TIMER("Attempt to move candidates");
        std::vector<BlockWeight> block_weight_deltas(_p_graph->k());
        Random &rand = Random::instance();
        for (const auto [node, to] : move_candidates) {
          const BlockID from = _p_graph->block(node);

          KASSERT(to < _p_graph->k());
          KASSERT(from < _p_graph->k());

          const double prob = 1.0 *
                              (_p_ctx->graph->max_block_weight(to) - _p_graph->block_weight(to)) /
                              weight_to_block[to];
          if (rand.random_bool(prob)) {
            _p_graph->set_block<false>(node, to);
            _marker.set(node);

            const NodeWeight weight = _p_graph->node_weight(node);
            block_weight_deltas[from] -= weight;
            block_weight_deltas[to] += weight;
          }
        }
        mpi::barrier(_p_graph->communicator());
        STOP_TIMER();

        TIMED_SCOPE("Synchronize partition state after fast rebalance round") {
          struct Message {
            NodeID node;
            BlockID block;
          };

          mpi::graph::sparse_alltoall_interface_to_pe<Message>(
              p_graph.graph(),
              [&](const NodeID u) -> bool { return _marker.get(u); },
              [&](const NodeID u) -> Message { return {.node = u, .block = p_graph.block(u)}; },
              [&](const auto &recv_buffer, const PEID pe) {
                tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
                  const auto [local_node_on_pe, block] = recv_buffer[i];
                  const auto global_node =
                      static_cast<GlobalNodeID>(p_graph.offset_n(pe) + local_node_on_pe);
                  const NodeID local_node = p_graph.global_to_local_node(global_node);
                  p_graph.set_block<false>(local_node, block);
                });
              }
          );

          MPI_Allreduce(
              MPI_IN_PLACE,
              block_weight_deltas.data(),
              p_graph.k(),
              mpi::type::get<BlockWeight>(),
              MPI_SUM,
              p_graph.communicator()
          );

          _p_graph->pfor_blocks([&](const BlockID b) {
            _p_graph->set_block_weight(b, _p_graph->block_weight(b) + block_weight_deltas[b]);
          });

          mpi::barrier(_p_graph->communicator());
        };

        // Variable names sound wronng, but are right
        DBG0 << "Fast balancing round improved L1 imbalance from " << current_imbalance_distance
             << " to " << metrics::imbalance_l1(p_graph, p_ctx);

        KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);

        // Reinit PQs @todo try to maintain valid PQ state
        TIMED_SCOPE("Reinit PQs") {
          DBG << "Reinit PQ, " << metrics::imbalance_l1(p_graph, p_ctx);
          init_pq();
          DBG << "Done";
          mpi::barrier(_p_graph->communicator());
        };
      }

      previous_imbalance_distance = metrics::imbalance_l1(p_graph, p_ctx);
    }
  }

  IFSTATS(_stats.final_feasible = metrics::is_feasible(p_graph, p_ctx));
  IFSTATS(_stats.final_cut = metrics::edge_cut(p_graph));
  IFSTATS(_stats.final_imbalance = metrics::imbalance(p_graph));
  IFSTATS(_stats.final_num_imbalanced_blocks = metrics::num_imbalanced_blocks(p_graph, p_ctx));
  IFSTATS(print_statistics());
}

void GreedyBalancer::print_candidates(
    const std::vector<MoveCandidate> &moves, const std::string &desc
) const {
  std::stringstream ss;
  ss << desc << " [";
  for (const auto &[node, from, to, weight, rel_gain] : moves) {
    ss << "{node=" << node << ", from=" << from << ", to=" << to << ", weight=" << weight
       << ", rel_gain=" << rel_gain << "}";
  }
  ss << "]";
  DLOG << "candidates=" << ss.str();
}

void GreedyBalancer::print_overloads() const {
  for (const BlockID b : _p_graph->blocks()) {
    LOG << V(b) << V(block_overload(b));
  }
}

void GreedyBalancer::perform_moves(const std::vector<MoveCandidate> &moves) {
  for (const auto &move : moves) {
    perform_move(move);
  }
}

void GreedyBalancer::perform_move(const MoveCandidate &move) {
  const auto &[node, from, to, weight, rel_gain] = move;

  if (from == to) { // Should only happen on root
    KASSERT(mpi::get_comm_rank(_p_graph->communicator()) == 0);
    return;
  }

  if (_p_graph->contains_global_node(node)) {
    const NodeID u = _p_graph->global_to_local_node(node);

    if (_p_graph->graph().is_owned_global_node(node)) { // Move node on this PE
      KASSERT(u < _p_graph->n());
      KASSERT(_pq.contains(u));

      _pq.remove(from, u);
      _pq_weight[from] -= weight;

      // Activate neighbors
      for (const NodeID v : _p_graph->adjacent_nodes(u)) {
        if (!_p_graph->is_owned_node(v)) {
          continue;
        }

        if (!_marker.get(v) && _p_graph->block(v) == from) {
          add_to_pq(from, v);
          _marker.set(v);
        }
      }
    }

    _p_graph->set_block(u, to);
  } else { // Only update block weight
    _p_graph->set_block_weight(from, _p_graph->block_weight(from) - weight);
    _p_graph->set_block_weight(to, _p_graph->block_weight(to) + weight);
  }
}

template <typename Elements>
Elements GreedyBalancer::reduce_buckets_or_move_candidates(Elements &&elements) {
  const int size = mpi::get_comm_size(_p_graph->communicator());
  const int rank = mpi::get_comm_rank(_p_graph->communicator());

  enum class Role {
    SENDER,
    RECEIVER,
    PAUSE
  };

  int active_size = size;
  while (active_size > 1) {
    if (rank >= active_size) {
      continue;
    }

    // false = receiver
    // true = sender
    const Role role = [&] {
      if (active_size % 2 == 0) {
        if (rank < active_size / 2) {
          return Role::RECEIVER;
        } else {
          return Role::SENDER;
        }
      } else {
        if (rank == 0) {
          return Role::PAUSE;
        }
        if (rank <= active_size / 2) {
          return Role::RECEIVER;
        } else {
          return Role::SENDER;
        }
      }
    }();

    if (role == Role::SENDER) {
      const int dest = rank - active_size / 2;
      mpi::send(elements.data(), elements.size(), dest, 0, _p_graph->communicator());
      return {};
    } else if (role == Role::RECEIVER) {
      using Value = typename Elements::value_type;
      const int src = rank + active_size / 2;
      Elements tmp_buffer = mpi::probe_recv<Value, Elements>(src, 0, _p_graph->communicator());
      if constexpr (std::is_same_v<Value, MoveCandidate>) {
        elements = reduce_move_candidates(std::move(elements), std::move(tmp_buffer));
      }
      if constexpr (std::is_same_v<Value, NodeWeight>) {
        elements = reduce_buckets(std::move(elements), std::move(tmp_buffer));
      }
    }

    active_size = active_size / 2 + active_size % 2;
  }

  return std::move(elements);
}

auto GreedyBalancer::reduce_move_candidates(
    std::vector<MoveCandidate> &&a, std::vector<MoveCandidate> &&b
) -> std::vector<MoveCandidate> {
  std::vector<MoveCandidate> ans;

  // Precondition: candidates are sorted by from block
  KASSERT([&] {
    for (std::size_t i = 1; i < a.size(); ++i) {
      KASSERT(a[i].from >= a[i - 1].from);
    }
    for (std::size_t i = 1; i < b.size(); ++i) {
      KASSERT(b[i].from >= b[i - 1].from);
    }
    return true;
  }());

  std::size_t i = 0; // index in a
  std::size_t j = 0; // index in b

  std::vector<NodeWeight> target_block_weight_delta(_p_ctx->k);

  for (i = 0, j = 0; i < a.size() && j < b.size();) {
    const BlockID from = std::min<BlockID>(a[i].from, b[j].from);

    // Find region in `a` and `b` with nodes from `from`
    std::size_t i_end = i;
    std::size_t j_end = j;
    while (i_end < a.size() && a[i_end].from == from) {
      ++i_end;
    }
    while (j_end < b.size() && b[j_end].from == from) {
      ++j_end;
    }

    // Pick best set of nodes
    const std::size_t num_in_a = i_end - i;
    const std::size_t num_in_b = j_end - j;
    const std::size_t num = num_in_a + num_in_b;

    std::vector<MoveCandidate> candidates(num);
    std::copy(a.begin() + i, a.begin() + i_end, candidates.begin());
    std::copy(b.begin() + j, b.begin() + j_end, candidates.begin() + num_in_a);
    std::sort(candidates.begin(), candidates.end(), [&](const auto &lhs, const auto &rhs) {
      return lhs.rel_gain > rhs.rel_gain || (lhs.rel_gain == rhs.rel_gain && lhs.node > rhs.node);
    });

    NodeWeight total_weight = 0;
    NodeID added_to_ans = 0;
    for (NodeID candidate = 0; candidate < candidates.size(); ++candidate) {
      const BlockID to = candidates[candidate].to;
      const NodeWeight weight = candidates[candidate].weight;

      // Only pick candidate if it does not overload the target block
      if (from != to && _p_graph->block_weight(to) + target_block_weight_delta[to] + weight >
                            _p_ctx->graph->max_block_weight(to)) {
        continue;
      }

      ans.push_back(candidates[candidate]);
      total_weight += weight;
      if (from != to) {
        target_block_weight_delta[to] += weight;
      }
      ++added_to_ans;

      // Only pick candidates while we do not have enough weight to balance the
      // block
      if (total_weight >= block_overload(from) ||
          added_to_ans >= _ctx.refinement.greedy_balancer.num_nodes_per_block) {
        break;
      }
    }

    // Move forward
    i = i_end;
    j = j_end;
  }

  // Keep remaining moves
  while (i < a.size()) {
    const BlockID from = a[i].from;
    const BlockID to = a[i].to;
    const NodeWeight weight = a[i].weight;

    if (from == to || _p_graph->block_weight(to) + target_block_weight_delta[to] + weight <=
                          _p_ctx->graph->max_block_weight(to)) {
      ans.push_back(a[i]);
      if (from != to) {
        target_block_weight_delta[to] += weight;
      }
    }

    ++i;
  }
  while (j < b.size()) {
    const BlockID from = b[j].from;
    const BlockID to = b[j].to;
    const NodeWeight weight = b[j].weight;

    if (from == to || _p_graph->block_weight(to) + target_block_weight_delta[to] + weight <=
                          _p_ctx->graph->max_block_weight(to)) {
      ans.push_back(b[j]);
      if (from != to) {
        target_block_weight_delta[to] += weight;
      }
    }

    ++j;
  }

  return ans;
}

NoinitVector<NodeWeight>
GreedyBalancer::reduce_buckets(NoinitVector<NodeWeight> &&a, NoinitVector<NodeWeight> &&b) {
  KASSERT(a.size() == b.size());
  tbb::parallel_for<std::size_t>(0, a.size(), [&](std::size_t i) { a[i] += b[i]; });
  return std::move(a);
}

auto GreedyBalancer::pick_move_candidates() -> std::vector<MoveCandidate> {
  std::vector<MoveCandidate> candidates;

  for (const BlockID from : _p_graph->blocks()) {
    if (block_overload(from) == 0) {
      continue;
    }

    // Fetch up to `num_nodes_per_block` move candidates from the PQ,
    // but keep them in the PQ, since they might not get moved
    NodeID num = 0;
    for (num = 0; num < _ctx.refinement.greedy_balancer.num_nodes_per_block; ++num) {
      if (_pq.empty(from)) {
        break;
      }

      const NodeID u = _pq.peek_max_id(from);
      const double relative_gain = _pq.peek_max_key(from);
      const NodeWeight u_weight = _p_graph->node_weight(u);
      _pq.pop_max(from);
      _pq_weight[from] -= u_weight;

      auto [to, actual_relative_gain] = compute_gain(u, from);

      if (relative_gain == actual_relative_gain) {
        MoveCandidate candidate{
            _p_graph->local_to_global_node(u), from, to, u_weight, actual_relative_gain};
        candidates.push_back(candidate);
        IFSTATS(++_stats.local_num_nonconflicts);
      } else {
        add_to_pq(from, u, u_weight, actual_relative_gain);
        --num; // Retry
        IFSTATS(++_stats.local_num_conflicts);
      }
    }

    for (NodeID rnum = 0; rnum < num; ++rnum) {
      KASSERT(candidates.size() > rnum);
      const auto &candidate = candidates[candidates.size() - rnum - 1];
      _pq.push(from, _p_graph->global_to_local_node(candidate.node), candidate.rel_gain);
      _pq_weight[from] += candidate.weight;
    }
  }

  return candidates;
}

void GreedyBalancer::init_pq() {
  const BlockID k = _p_graph->k();

  tbb::enumerable_thread_specific<std::vector<DynamicBinaryMinHeap<NodeID, double>>> local_pq_ets{
      [&] {
        return std::vector<DynamicBinaryMinHeap<NodeID, double>>(k);
      }};

  tbb::enumerable_thread_specific<std::vector<NodeWeight>> local_pq_weight_ets{[&] {
    return std::vector<NodeWeight>(k);
  }};

  _marker.reset();
  if (_marker.capacity() < _p_graph->n()) {
    _marker.resize(_p_graph->n());
  }

  // Build thread-local PQs: one PQ for each thread and block, each PQ for block
  // b has at most roughly |overload[b]| weight
  tbb::parallel_for(static_cast<NodeID>(0), _p_graph->n(), [&](const NodeID u) {
    auto &pq = local_pq_ets.local();
    auto &pq_weight = local_pq_weight_ets.local();

    const BlockID b = _p_graph->block(u);
    const BlockWeight overload = block_overload(b);

    if (overload > 0) { // Node in overloaded block
      const auto [max_gainer, rel_gain] = compute_gain(u, b);
      const bool need_more_nodes = (pq_weight[b] < overload);
      if (need_more_nodes || pq[b].empty() || rel_gain > pq[b].peek_key()) {
        if (!need_more_nodes) {
          const NodeWeight u_weight = _p_graph->node_weight(u);
          const NodeWeight min_weight = _p_graph->node_weight(pq[b].peek_id());
          if (pq_weight[b] + u_weight - min_weight >= overload) {
            pq[b].pop();
          }
        }
        pq[b].push(u, rel_gain);
        _marker.set(u);
      }
    }
  });

  // Build global PQ: one PQ per block, block-level parallelism
  _pq.clear();
  if (_pq.capacity() < _p_graph->n()) {
    _pq = DynamicBinaryMinMaxForest<NodeID, double>(_p_graph->n(), _ctx.partition.k);
  }

  tbb::parallel_for(static_cast<BlockID>(0), k, [&](const BlockID b) {
    _pq_weight[b] = 0;

    for (auto &pq : local_pq_ets) {
      for (const auto &[u, rel_gain] : pq[b].elements()) {
        add_to_pq(b, u, _p_graph->node_weight(u), rel_gain);
      }
    }
  });
}

void GreedyBalancer::init_buckets() {
  _buckets.resize(kBucketsPerBlock * _p_graph->k());
  tbb::parallel_for<std::size_t>(0, _buckets.size(), [&](const std::size_t index) {
    _buckets[index] = 0;
  });

  tbb::parallel_for(static_cast<NodeID>(0), _p_graph->n(), [&](const NodeID node) {
    const BlockID block = _p_graph->block(node);
    if (block_overload(block) > 0) { // Node in overloaded block
      const auto [max_gainer, rel_gain] = compute_gain(node, block);
      const auto bucket = get_bucket(rel_gain);
      get_bucket_value(block, bucket) += _p_graph->node_weight(node);
    }
  });
}

std::pair<BlockID, double>
GreedyBalancer::compute_gain(const NodeID u, const BlockID u_block) const {
  const NodeWeight u_weight = _p_graph->node_weight(u);
  BlockID max_gainer = u_block;
  EdgeWeight max_external_gain = 0;
  EdgeWeight internal_degree = 0;

  auto action = [&](auto &map) {
    // Compute external degree to each adjacent block that can take u without
    // becoming overloaded
    for (const auto [e, v] : _p_graph->neighbors(u)) {
      const BlockID v_block = _p_graph->block(v);
      if (u_block != v_block &&
          _p_graph->block_weight(v_block) + u_weight <= _p_ctx->graph->max_block_weight(v_block)) {
        map[v_block] += _p_graph->edge_weight(e);
      } else if (u_block == v_block) {
        internal_degree += _p_graph->edge_weight(e);
      }
    }

    // Select neighbor that maximizes gain
    auto &rand = Random::instance();
    for (const auto [block, gain] : map.entries()) {
      if (gain > max_external_gain || (gain == max_external_gain && rand.random_bool())) {
        max_gainer = block;
        max_external_gain = gain;
      }
    }
    map.clear();
  };

  auto &rating_map = _rating_map.local();
  rating_map.update_upper_bound_size(_p_graph->degree(u));
  rating_map.run_with_map(action, action);

  // Compute absolute and relative gain based on internal degree / external gain
  const EdgeWeight gain = max_external_gain - internal_degree;
  const double relative_gain = compute_relative_gain(gain, u_weight);
  return {max_gainer, relative_gain};
}

BlockWeight GreedyBalancer::block_overload(const BlockID b) const {
  static_assert(
      std::numeric_limits<BlockWeight>::is_signed,
      "This must be changed when using an unsigned data type for "
      "block weights!"
  );
  KASSERT(b < _p_graph->k());
  return std::max<BlockWeight>(0, _p_graph->block_weight(b) - _p_ctx->graph->max_block_weight(b));
}

double GreedyBalancer::compute_relative_gain(
    const EdgeWeight absolute_gain, const NodeWeight weight
) const {
  if (absolute_gain >= 0) {
    return absolute_gain * weight;
  } else {
    return 1.0 * absolute_gain / weight;
  }
}

bool GreedyBalancer::add_to_pq(const BlockID b, const NodeID u) {
  KASSERT(b == _p_graph->block(u));

  const auto [to, rel_gain] = compute_gain(u, b);
  return add_to_pq(b, u, _p_graph->node_weight(u), rel_gain);
}

bool GreedyBalancer::add_to_pq(
    const BlockID b, const NodeID u, const NodeWeight u_weight, const double rel_gain
) {
  KASSERT(u_weight == _p_graph->node_weight(u));
  KASSERT(b == _p_graph->block(u));

  if (_pq_weight[b] < block_overload(b) || _pq.empty(b) || rel_gain > _pq.peek_min_key(b)) {
    _pq.push(b, u, rel_gain);
    _pq_weight[b] += u_weight;

    if (rel_gain > _pq.peek_min_key(b)) {
      const NodeID min_node = _pq.peek_min_id(b);
      const NodeWeight min_weight = _p_graph->node_weight(min_node);
      if (_pq_weight[b] - min_weight >= block_overload(b)) {
        _pq.pop_min(b);
        _pq_weight[b] -= min_weight;
      }
    }

    return true;
  }

  return false;
}

void GreedyBalancer::reset_statistics() {
  _stats = {};
}

void GreedyBalancer::print_statistics() const {
  const GlobalNodeID global_num_conflicts =
      mpi::allreduce(_stats.local_num_conflicts, MPI_SUM, _p_graph->communicator());
  const GlobalNodeID global_num_nonconflicts =
      mpi::allreduce(_stats.local_num_nonconflicts, MPI_SUM, _p_graph->communicator());

  STATS << "GreedyBalancer:";
  STATS << "  * Feasible changed: " << C(_stats.initial_feasible, _stats.final_feasible);
  STATS << "  * Number of rounds: " << _stats.num_reduction_rounds;
  STATS << "  * Change in imbalance: " << C(_stats.initial_imbalance, _stats.final_imbalance);
  STATS << "  * Change in number of imbalanced blocks: "
        << C(_stats.initial_num_imbalanced_blocks, _stats.final_num_imbalanced_blocks) << " = by "
        << _stats.initial_num_imbalanced_blocks - _stats.final_num_imbalanced_blocks;
  STATS << "  * Change in edge cut: " << C(_stats.initial_cut, _stats.final_cut) << " = by "
        << _stats.initial_cut - _stats.final_cut;
  STATS << "  * Number of moved nodes: "
        << _stats.num_adjacent_moves + _stats.num_nonadjacent_moves;
  STATS << "    # of moves to adjacent blocks: " << _stats.num_adjacent_moves;
  STATS << "    # of moves to nonadjacent blocks: " << _stats.num_nonadjacent_moves;
  STATS << "    # of conflicts: " << global_num_conflicts
        << " (= " << 100.0 * global_num_conflicts / (global_num_conflicts + global_num_nonconflicts)
        << "% of selected nodes)";
}

NoinitVector<NodeWeight> GreedyBalancer::compactify_buckets() const {
  const BlockID num_overloaded_blocks = metrics::num_imbalanced_blocks(*_p_graph, *_p_ctx);
  NoinitVector<NodeWeight> compact(kBucketsPerBlock * num_overloaded_blocks);

  BlockID current_overloaded_block = 0;
  for (const BlockID block : _p_graph->blocks()) {
    if (_p_graph->block_weight(block) > _p_ctx->graph->max_block_weight(block)) {
      for (int bucket = 0; bucket < kBucketsPerBlock; ++bucket) {
        const std::size_t index = current_overloaded_block * kBucketsPerBlock + bucket;
        KASSERT(index < compact.size());
        compact[index] = get_bucket_value(block, bucket);
      }
      current_overloaded_block++;
    }
  }

  return compact;
}
} // namespace kaminpar::dist
