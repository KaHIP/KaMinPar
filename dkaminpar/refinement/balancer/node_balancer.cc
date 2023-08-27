/*******************************************************************************
 * @file:   greedy_balancer.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#include "dkaminpar/refinement/balancer/node_balancer.h"

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/graphutils/synchronization.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/binary_reduction_tree.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/refinement/balancer/reductions.h"

#include "common/math.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
NodeBalancerFactory::NodeBalancerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
NodeBalancerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<NodeBalancer>(_ctx, p_graph, p_ctx);
}

NodeBalancer::NodeBalancer(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _p_graph(p_graph),
      _ctx(ctx),
      _nb_ctx(ctx.refinement.node_balancer),
      _p_ctx(p_ctx),
      _pq(ctx.partition.graph->n, ctx.partition.k),
      _pq_weight(ctx.partition.k),
      _marker(ctx.partition.graph->n),
      _buckets(
          p_graph, p_ctx, _nb_ctx.par_enable_positive_gain_buckets, _nb_ctx.par_gain_bucket_base
      ),
      _cached_cutoff_buckets(_p_graph.k()),
      _gain_calculator(p_graph) {}

void NodeBalancer::initialize() {
  // Only initialize the balancer is the partition is actually imbalanced
  if (metrics::is_feasible(_p_graph, _p_ctx)) {
    return;
  }

  // Allocate _marker memory
  _marker.reset();
  if (_marker.capacity() < _p_graph.n()) {
    _marker.resize(_p_graph.n());
  }

  // Allocate helper PQs
  tbb::enumerable_thread_specific<std::vector<DynamicBinaryMinHeap<NodeID, double>>> local_pq_ets{
      [&] {
        return std::vector<DynamicBinaryMinHeap<NodeID, double>>(_p_graph.k());
      }};
  tbb::enumerable_thread_specific<std::vector<NodeWeight>> local_pq_weight_ets{[&] {
    return std::vector<NodeWeight>(_p_graph.k());
  }};

  // Build thread-local PQs: one PQ for each thread and block, each PQ for block
  // b has at most roughly |overload[b]| weight
  tbb::parallel_for(static_cast<NodeID>(0), _p_graph.n(), [&](const NodeID u) {
    auto &pq = local_pq_ets.local();
    auto &pq_weight = local_pq_weight_ets.local();

    const BlockID from = _p_graph.block(u);
    const BlockWeight overload = block_overload(from);

    if (overload > 0) { // Node in overloaded block
      const auto [gain, max_gainer] = _gain_calculator.compute_relative_gain(u, _p_ctx);
      const bool need_more_nodes = (pq_weight[from] < overload);
      if (need_more_nodes || pq[from].empty() || gain > pq[from].peek_key()) {
        if (!need_more_nodes) {
          const NodeWeight u_weight = _p_graph.node_weight(u);
          const NodeWeight min_weight = _p_graph.node_weight(pq[from].peek_id());
          if (pq_weight[from] + u_weight - min_weight >= overload) {
            pq[from].pop();
          }
        }
        pq[from].push(u, gain);
        _marker.set(u);
      }
    }
  });

  // Build global PQ: one PQ per block, block-level parallelism
  _pq.clear();
  if (_pq.capacity() < _p_graph.n()) {
    _pq = DynamicBinaryMinMaxForest<NodeID, double>(_p_graph.n(), _ctx.partition.k);
  }

  _p_graph.pfor_blocks([&](const BlockID block) {
    _pq_weight[block] = 0;

    for (auto &pq : local_pq_ets) {
      for (const auto &[u, rel_gain] : pq[block].elements()) {
        try_pq_insertion(block, u, _p_graph.node_weight(u), rel_gain);
      }
    }
  });
}

bool NodeBalancer::refine() {
  SCOPED_TIMER("Balancer");

  // Only balance the partition if it is infeasible
  if (metrics::is_feasible(_p_graph, _p_ctx)) {
    return false;
  }

  const PEID size = mpi::get_comm_size(_p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  double previous_imbalance_distance =
      is_sequential_balancing_enabled() ? metrics::imbalance_l1(_p_graph, _p_ctx) : 0.0;

  for (int round = 0; round < _nb_ctx.max_num_rounds; round++) {
    if (metrics::is_feasible(_p_graph, _p_ctx)) {
      break;
    }

    if (is_sequential_balancing_enabled()) {
      if (!perform_sequential_round()) {
        DBG0 << "terminated by sequential round";
        break;
      }
    }

    if (is_parallel_balancing_enabled()) {
      const double current_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
      DBG0 << "Sequential rebalancing changed imbalance: " << previous_imbalance_distance << " --> "
           << current_imbalance_distance << " = by "
           << (previous_imbalance_distance - current_imbalance_distance) /
                  previous_imbalance_distance
           << "; threshold: " << _ctx.refinement.node_balancer.par_threshold;

      if ((previous_imbalance_distance - current_imbalance_distance) / previous_imbalance_distance <
              _nb_ctx.par_threshold ||
          !is_sequential_balancing_enabled()) {
        mpi::barrier(_p_graph.communicator());

        if (!perform_parallel_round()) {
          DBG0 << "terminated by parallel round";
          break;
        }

        const double next_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
        if (previous_imbalance_distance == next_imbalance_distance) {
          DBG0 << "Parallel rebalancing did not improve imbalance: switching to sequential "
                  "rebalancing only";
          _stalled = true;
        }
        previous_imbalance_distance = next_imbalance_distance;
      }
    }
  }

  return false;
}

bool NodeBalancer::perform_sequential_round() {
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  START_TIMER("Pick and reduce move candidates");
  auto candidates = reduce_candidates(
      pick_sequential_candidates(),
      _ctx.refinement.node_balancer.seq_num_nodes_per_block,
      _p_graph,
      _p_ctx
  );
  STOP_TIMER();

  START_TIMER("Perform moves on root PE");
  if (rank == 0) {
    // Move nodes that already have a target block
    for (const auto &move : candidates) {
      if (move.from != move.to) {
        perform_move(move, true);
      }
    }

    // Move nodes that do not have a target block
    BlockID cur = 0;
    for (auto &candidate : candidates) {
      auto &[node, from, to, weight, rel_gain] = candidate;

      if (from == to) {
        // Look for next block that can take node
        while (cur == from ||
               _p_graph.block_weight(cur) + weight > _p_ctx.graph->max_block_weight(cur)) {
          ++cur;
          if (cur >= _p_ctx.k) {
            cur = 0;
          }
        }

        to = cur;
        perform_move(candidate, true);
      }
    }
  }
  STOP_TIMER();

  // Broadcast winners
  START_TIMER("Broadcast reduction result");
  const std::size_t num_winners = mpi::bcast(candidates.size(), 0, _p_graph.communicator());
  candidates.resize(num_winners);
  mpi::bcast(candidates.data(), num_winners, 0, _p_graph.communicator());
  STOP_TIMER();

  START_TIMER("Perform moves");
  if (rank != 0) {
    perform_moves(candidates, true);
  }
  STOP_TIMER();

  KASSERT(
      graph::debug::validate_partition(_p_graph),
      "balancer produced invalid partition",
      assert::heavy
  );

  return num_winners > 0;
}

void NodeBalancer::perform_moves(
    const std::vector<Candidate> &moves, const bool update_block_weights
) {
  for (const auto &move : moves) {
    perform_move(move, update_block_weights);
  }
}

void NodeBalancer::perform_move(const Candidate &move, const bool update_block_weights) {
  const auto &[node, from, to, weight, rel_gain] = move;

  if (from == to) { // Should only happen on root
    KASSERT(mpi::get_comm_rank(_p_graph.communicator()) == 0);
    return;
  }

  if (_p_graph.contains_global_node(node)) {
    const NodeID u = _p_graph.global_to_local_node(node);

    if (_p_graph.is_owned_global_node(node)) { // Move node on this PE
      KASSERT(u < _p_graph.n());
      KASSERT(_pq.contains(u));

      _pq.remove(from, u);
      _pq_weight[from] -= weight;

      // Activate neighbors
      for (const NodeID v : _p_graph.adjacent_nodes(u)) {
        if (!_p_graph.is_owned_node(v)) {
          continue;
        }

        if (!_marker.get(v) && _p_graph.block(v) == from) {
          try_pq_insertion(from, v);
          _marker.set(v);
        }
      }
    }

    if (update_block_weights) {
      _p_graph.set_block(u, to);
    } else {
      _p_graph.set_block<false>(u, to);
    }
  } else if (update_block_weights) { // Only update block weight
    _p_graph.set_block_weight(from, _p_graph.block_weight(from) - weight);
    _p_graph.set_block_weight(to, _p_graph.block_weight(to) + weight);
  }
}

std::vector<NodeBalancer::Candidate> NodeBalancer::pick_sequential_candidates() {
  std::vector<Candidate> candidates;

  for (const BlockID from : _p_graph.blocks()) {
    if (block_overload(from) == 0) {
      continue;
    }

    // Fetch up to `num_nodes_per_block` move candidates from the PQ,
    // but keep them in the PQ, since they might not get moved
    NodeID num = 0;
    for (num = 0; num < _nb_ctx.seq_num_nodes_per_block; ++num) {
      if (_pq.empty(from)) {
        break;
      }

      const NodeID u = _pq.peek_max_id(from);
      const double relative_gain = _pq.peek_max_key(from);
      const NodeWeight u_weight = _p_graph.node_weight(u);
      _pq.pop_max(from);
      _pq_weight[from] -= u_weight;

      auto [actual_relative_gain, to] = _gain_calculator.compute_relative_gain(u, _p_ctx);

      if (relative_gain == actual_relative_gain) {
        Candidate candidate{
            _p_graph.local_to_global_node(u), from, to, u_weight, actual_relative_gain};
        candidates.push_back(candidate);
      } else {
        try_pq_insertion(from, u, u_weight, actual_relative_gain);
        --num; // Retry
      }
    }

    for (NodeID rnum = 0; rnum < num; ++rnum) {
      KASSERT(candidates.size() > rnum);
      const auto &candidate = candidates[candidates.size() - rnum - 1];
      _pq.push(from, _p_graph.global_to_local_node(candidate.id), candidate.gain);
      _pq_weight[from] += candidate.weight;
    }
  }

  return candidates;
}

BlockWeight NodeBalancer::block_overload(const BlockID block) const {
  static_assert(
      std::numeric_limits<BlockWeight>::is_signed,
      "This must be changed when using an unsigned data type for "
      "block weights!"
  );
  KASSERT(block < _p_graph.k());
  return std::max<BlockWeight>(
      0, _p_graph.block_weight(block) - _p_ctx.graph->max_block_weight(block)
  );
}

BlockWeight NodeBalancer::block_underload(const BlockID block) const {
  static_assert(
      std::numeric_limits<BlockWeight>::is_signed,
      "This must be changed when using an unsigned data type for "
      "block weights!"
  );
  KASSERT(block < _p_graph.k());
  return std::max<BlockWeight>(
      0, _p_ctx.graph->max_block_weight(block) - _p_graph.block_weight(block)
  );
}

bool NodeBalancer::try_pq_insertion(const BlockID b, const NodeID u) {
  KASSERT(b == _p_graph.block(u));

  const auto [rel_gain, to] = _gain_calculator.compute_relative_gain(u, _p_ctx);
  return try_pq_insertion(b, u, _p_graph.node_weight(u), rel_gain);
}

bool NodeBalancer::try_pq_insertion(
    const BlockID b, const NodeID u, const NodeWeight u_weight, const double rel_gain
) {
  KASSERT(u_weight == _p_graph.node_weight(u));
  KASSERT(b == _p_graph.block(u));

  if (_pq_weight[b] < block_overload(b) || _pq.empty(b) || rel_gain > _pq.peek_min_key(b)) {
    _pq.push(b, u, rel_gain);
    _pq_weight[b] += u_weight;

    if (rel_gain > _pq.peek_min_key(b)) {
      const NodeID min_node = _pq.peek_min_id(b);
      const NodeWeight min_weight = _p_graph.node_weight(min_node);
      if (_pq_weight[b] - min_weight >= block_overload(b)) {
        _pq.pop_min(b);
        _pq_weight[b] -= min_weight;
      }
    }

    return true;
  }

  return false;
}

bool NodeBalancer::perform_parallel_round() {
  SCOPED_TIMER("Fast rebalancing");

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  START_TIMER("Computing weight buckets");
  _buckets.clear();
  for (const BlockID from : _p_graph.blocks()) {
    for (const auto &[node, pq_gain] : _pq.elements(from)) {
      KASSERT(_p_graph.block(node) == from);
      const auto [actual_gain, to] = _gain_calculator.compute_relative_gain(node, _p_ctx);
      _buckets.add(from, _p_graph.node_weight(node), actual_gain);
    }
  }
  const auto &cutoff_buckets =
      _buckets.compute_cutoff_buckets(reduce_buckets_mpireduce(_buckets, _p_graph));
  STOP_TIMER();

  // Find move candidates
  std::vector<Candidate> candidates;
  std::vector<BlockWeight> block_weight_deltas(_p_graph.k());

  START_TIMER("Find move candidates");
  for (const BlockID from : _p_graph.blocks()) {
    for (const auto &[node, pq_gain] : _pq.elements(from)) {
      // @todo avoid double gain recomputation
      const auto [actual_gain, to] = _gain_calculator.compute_relative_gain(node, _p_ctx);
      const auto bucket = _buckets.compute_bucket(actual_gain);

      if (bucket < cutoff_buckets[from]) {
        Candidate candidate = {
            .id = _p_graph.local_to_global_node(node),
            .from = from,
            .to = to,
            .weight = _p_graph.node_weight(node),
            .gain = actual_gain,
        };

        if (candidate.from == candidate.to) {
          [[maybe_unused]] const bool reassigned =
              assign_feasible_target_block(candidate, block_weight_deltas);
          KASSERT(reassigned);
        }

        block_weight_deltas[candidate.to] += candidate.weight;
        candidates.push_back(candidate);
      }
    }
  }
  MPI_Barrier(_p_graph.communicator());
  STOP_TIMER();

  // Compute total weight to each block
  START_TIMER("Allreduce weight to block");
  MPI_Allreduce(
      MPI_IN_PLACE,
      block_weight_deltas.data(),
      asserting_cast<int>(_p_graph.k()),
      mpi::type::get<BlockWeight>(),
      MPI_SUM,
      _p_graph.communicator()
  );
  STOP_TIMER();

  // Perform moves
  START_TIMER("Attempt to move candidates");
  Random &rand = Random::instance();

  std::size_t num_rejected_candidates;
  std::vector<BlockWeight> actual_block_weight_deltas;
  bool balanced_moves = false;

  for (int attempt = 0;
       !balanced_moves && attempt < std::max<int>(1, _nb_ctx.par_num_dicing_attempts);
       ++attempt) {
    num_rejected_candidates = 0;
    actual_block_weight_deltas.clear();
    actual_block_weight_deltas.resize(_p_graph.k());

    for (std::size_t i = 0; i < candidates.size() - num_rejected_candidates; ++i) {
      const auto &candidate = candidates[i];
      const double probability =
          1.0 * block_underload(candidate.to) / block_weight_deltas[candidate.to];
      if (rand.random_bool(probability)) {
        actual_block_weight_deltas[candidate.to] += candidate.weight;
        actual_block_weight_deltas[candidate.from] -= candidate.weight;
      } else {
        ++num_rejected_candidates;
        std::swap(candidates[i], candidates[candidates.size() - num_rejected_candidates]);
        --i;
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

    // Check that the moves do not overload a previously non-overloaded block
    balanced_moves = true;
    for (const BlockID block : _p_graph.blocks()) {
      if (block_overload(block) == 0 &&
          block_underload(block) < actual_block_weight_deltas[block]) {
        balanced_moves = false;
        break;
      }
    }
  }
  STOP_TIMER();

  if (balanced_moves || _nb_ctx.par_accept_imbalanced_moves) {
    for (const BlockID block : _p_graph.blocks()) {
      _p_graph.set_block_weight(
          block, _p_graph.block_weight(block) + actual_block_weight_deltas[block]
      );
    }

    candidates.resize(candidates.size() - num_rejected_candidates);
    perform_moves(candidates, false);
  }

  TIMED_SCOPE("Synchronize partition state after fast rebalance round") {
    struct Message {
      NodeID node;
      BlockID block;
    };

    // @todo don't iterate over the entire graph
    mpi::graph::sparse_alltoall_interface_to_pe<Message>(
        _p_graph.graph(),
        [&](const NodeID u) -> bool { return _marker.get(u); },
        [&](const NodeID u) -> Message { return {.node = u, .block = _p_graph.block(u)}; },
        [&](const auto &recv_buffer, const PEID pe) {
          tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
            const auto [their_lnode, to] = recv_buffer[i];
            const NodeID lnode = _p_graph.map_foreign_node(their_lnode, pe);
            _p_graph.set_block<false>(lnode, to);
          });
        }
    );
  };

  return true;
}

bool NodeBalancer::is_sequential_balancing_enabled() const {
  return _stalled || _nb_ctx.enable_parallel_balancing;
}

bool NodeBalancer::is_parallel_balancing_enabled() const {
  return !_stalled && _nb_ctx.enable_sequential_balancing;
}

bool NodeBalancer::assign_feasible_target_block(
    Candidate &candidate, const std::vector<BlockWeight> &deltas
) const {
  do {
    ++candidate.to;
    if (candidate.to >= _p_ctx.k) {
      candidate.to = 0;
    }
  } while (candidate.from != candidate.to &&
           block_underload(candidate.to) < candidate.weight + deltas[candidate.to]);

  return candidate.from != candidate.to;
}
} // namespace kaminpar::dist
