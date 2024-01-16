/*******************************************************************************
 * Distributed balancing algorithm that moves individual nodes.
 *
 * @file:   greedy_balancer.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 ******************************************************************************/
#include "kaminpar-dist/refinement/balancer/node_balancer.h"

#include "kaminpar-mpi/binary_reduction_tree.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/graphutils/synchronization.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/refinement/balancer/reductions.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-common/math.h"
#include "kaminpar-common/random.h"

#define HEAVY assert::heavy

namespace kaminpar::dist {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

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
      _pq(p_graph.n(), p_graph.k()),
      _pq_weight(p_graph.k()),
      _marker(p_graph.n()),
      _buckets(
          p_graph, p_ctx, _nb_ctx.par_enable_positive_gain_buckets, _nb_ctx.par_gain_bucket_base
      ),
      _cached_cutoff_buckets(_p_graph.k()),
      _gain_calculator(_p_ctx.k),
      _target_blocks(_p_graph.n()),
      _tmp_gains(!_nb_ctx.par_update_pq_gains * _p_graph.n()) {
  _gain_calculator.init(_p_graph);
}

void NodeBalancer::initialize() {
  TIMER_BARRIER(_p_graph.communicator());
  SCOPED_TIMER("Node balancer");

  START_TIMER("Initialization");
  reinit();
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());
}

void NodeBalancer::reinit() {
  // debug::print_local_graph_stats(_p_graph.graph());

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
    if (_p_graph.degree(u) > _nb_ctx.par_high_degree_insertion_threshold) {
      return;
    }

    auto &pq = local_pq_ets.local();
    auto &pq_weight = local_pq_weight_ets.local();

    const BlockID from = _p_graph.block(u);
    const BlockWeight overload = block_overload(from);

    if (overload > 0) { // Node in overloaded block
      const auto max_gainer = _gain_calculator.compute_max_gainer(u, _p_ctx);
      const double rel_gain = max_gainer.relative_gain();
      _target_blocks[u] = max_gainer.block;

      const bool need_more_nodes = (pq_weight[from] < overload);
      if (need_more_nodes || pq[from].empty() || rel_gain > pq[from].peek_key()) {
        if (!need_more_nodes) {
          const NodeWeight u_weight = _p_graph.node_weight(u);
          const NodeWeight min_weight = _p_graph.node_weight(pq[from].peek_id());
          if (pq_weight[from] + u_weight - min_weight >= overload) {
            pq[from].pop();
          }
        }
        pq[from].push(u, rel_gain);
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

  _stalled = false;
}

bool NodeBalancer::refine() {
  TIMER_BARRIER(_p_graph.communicator());
  SCOPED_TIMER("Node balancer");

  // Only balance the partition if it is infeasible
  if (metrics::is_feasible(_p_graph, _p_ctx)) {
    return false;
  }

  KASSERT(debug::validate_partition(_p_graph), "invalid partition before balancing", HEAVY);

  const PEID size = mpi::get_comm_size(_p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  double previous_imbalance_distance =
      is_sequential_balancing_enabled() ? metrics::imbalance_l1(_p_graph, _p_ctx) : 0.0;

  for (int round = 0; round < _nb_ctx.max_num_rounds; round++) {
    TIMER_BARRIER(_p_graph.communicator());
    DBG0 << "Starting rebalancing round " << round << " of (at most) " << _nb_ctx.max_num_rounds;

    if (metrics::is_feasible(_p_graph, _p_ctx)) {
      DBG0 << "Partition is feasible ==> terminating";
      break;
    }

    if (is_sequential_balancing_enabled() && !perform_sequential_round()) {
      if (!_stalled) {
        DBG0 << "Sequential round stalled: switch to stalled mode";
        switch_to_stalled();
        continue;
      } else {
        DBG0 << "Terminated by sequential round";
        break;
      }
    }

    if (is_parallel_balancing_enabled()) {
      const double current_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
      const double seq_rebalance_rate =
          (previous_imbalance_distance - current_imbalance_distance) / previous_imbalance_distance;

      DBG0 << "Sequential rebalancing changed imbalance: " << previous_imbalance_distance << " --> "
           << current_imbalance_distance << " = by " << seq_rebalance_rate
           << "; threshold: " << _ctx.refinement.node_balancer.par_threshold;

      if (seq_rebalance_rate < _nb_ctx.par_threshold || !is_sequential_balancing_enabled()) {
        if (!perform_parallel_round(round)) {
          DBG0 << "Parallel round stalled: switch to stalled mode";
          switch_to_stalled();
          continue;
        }

        const double next_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
        [[maybe_unused]] const double par_rebalance_rate =
            (current_imbalance_distance - next_imbalance_distance) / current_imbalance_distance;
        DBG0 << "Parallel rebalancing changed imbalance: " << current_imbalance_distance << " --> "
             << next_imbalance_distance << " = by " << par_rebalance_rate;

        if (current_imbalance_distance == next_imbalance_distance) {
          DBG0 << "Parallel round stalled: switch to stalled mode";
          switch_to_stalled();
          // no continue -> update previous_imbalance_distance
        }

        previous_imbalance_distance = next_imbalance_distance;
      } else {
        previous_imbalance_distance = current_imbalance_distance;
      }
    }

    KASSERT(debug::validate_partition(_p_graph), "invalid partition after balancing round", HEAVY);
  }

  KASSERT(debug::validate_partition(_p_graph), "invalid partition after balancing", HEAVY);
  return false;
}

void NodeBalancer::switch_to_stalled() {
  TIMER_BARRIER(_p_graph.communicator());

  _stalled = true;

  // Reinit the balancer to fix blocks that were not overloaded in the beginning, but are
  // overloaded now due to imbalanced parallel moves
  START_TIMER("Reinitialize");
  reinit();
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());
}

bool NodeBalancer::perform_sequential_round() {
  TIMER_BARRIER(_p_graph.communicator());
  SCOPED_TIMER("Sequential round");

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  START_TIMER("Pick and reduce move candidates");
  auto candidates = reduce_candidates(
      pick_sequential_candidates(),
      _ctx.refinement.node_balancer.seq_num_nodes_per_block,
      _p_graph,
      _p_ctx
  );
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

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
  TIMER_BARRIER(_p_graph.communicator());

  // Broadcast winners
  START_TIMER("Broadcast winners");
  const std::size_t num_winners = mpi::bcast(candidates.size(), 0, _p_graph.communicator());
  candidates.resize(num_winners);
  mpi::bcast(candidates.data(), num_winners, 0, _p_graph.communicator());
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

  START_TIMER("Perform moves");
  if (rank != 0) {
    perform_moves(candidates, true);
  }
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

  KASSERT(debug::validate_partition(_p_graph), "balancer produced invalid partition", HEAVY);

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

      const auto max_gainer = _gain_calculator.compute_max_gainer(u, _p_ctx);
      const double actual_relative_gain = max_gainer.relative_gain();
      const BlockID to = max_gainer.block;
      _target_blocks[u] = to;

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

bool NodeBalancer::try_pq_insertion(const BlockID b_u, const NodeID u) {
  KASSERT(b_u == _p_graph.block(u));

  if (_p_graph.degree(u) > _nb_ctx.par_high_degree_insertion_threshold) {
    return false;
  }

  const auto max_gainer = _gain_calculator.compute_max_gainer(u, _p_ctx);
  _target_blocks[u] = max_gainer.block;
  return try_pq_insertion(b_u, u, _p_graph.node_weight(u), max_gainer.relative_gain());
}

bool NodeBalancer::try_pq_insertion(
    const BlockID b_u, const NodeID u, const NodeWeight w_u, const double rel_gain
) {
  KASSERT(w_u == _p_graph.node_weight(u));
  KASSERT(b_u == _p_graph.block(u));

  if (_p_graph.degree(u) > _nb_ctx.par_high_degree_insertion_threshold) {
    return false;
  }

  if (_pq_weight[b_u] < block_overload(b_u) || _pq.empty(b_u) || rel_gain > _pq.peek_min_key(b_u)) {
    _pq.push(b_u, u, rel_gain);
    _pq_weight[b_u] += w_u;

    if (rel_gain > _pq.peek_min_key(b_u)) {
      const NodeID min_node = _pq.peek_min_id(b_u);
      const NodeWeight min_weight = _p_graph.node_weight(min_node);
      if (_pq_weight[b_u] - min_weight >= block_overload(b_u)) {
        _pq.pop_min(b_u);
        _pq_weight[b_u] -= min_weight;
      }
    }

    return true;
  }

  return false;
}

bool NodeBalancer::perform_parallel_round(const int round) {
  TIMER_BARRIER(_p_graph.communicator());
  SCOPED_TIMER("Parallel round");

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  // Postpone PQ updates until after the iteration
  std::vector<std::tuple<BlockID, NodeID, double>> pq_updates;

  START_TIMER("Computing weight buckets");
  _buckets.clear();
  for (const BlockID from : _p_graph.blocks()) {
    for (const auto &[node, pq_gain] : _pq.elements(from)) {
      KASSERT(_p_graph.block(node) == from);

      // For high-degree nodes, assume that the PQ gain is up-to-date and skip recomputation
      if (_p_graph.degree(node) > _nb_ctx.par_high_degree_update_thresold &&
          ((round + 1) % _nb_ctx.par_high_degree_update_interval) == 0) {
        _buckets.add(from, _p_graph.node_weight(node), pq_gain);
        if (!_nb_ctx.par_update_pq_gains) {
          _tmp_gains[node] = pq_gain;
        }
        continue;
      }

      // For low-degree nodes, recalculate gain and update PQ
      const auto max_gainer = _gain_calculator.compute_max_gainer(node, _p_ctx);
      const double actual_gain = max_gainer.relative_gain();
      const BlockID to = max_gainer.block;

      if (_nb_ctx.par_update_pq_gains && pq_gain != actual_gain) {
        pq_updates.emplace_back(from, node, actual_gain);
      } else if (!_nb_ctx.par_update_pq_gains) {
        _tmp_gains[node] = actual_gain;
      }

      _buckets.add(from, _p_graph.node_weight(node), actual_gain);
      _target_blocks[node] = to;
    }
  }
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

  START_TIMER("Apply PQ updates");
  for (const auto &[from, node, gain] : pq_updates) {
    _pq.change_priority(from, node, gain);
  }
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

  START_TIMER("Computing cut-off buckets");
  const auto &cutoff_buckets =
      _buckets.compute_cutoff_buckets(reduce_buckets_mpireduce(_buckets, _p_graph));
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

  // Find move candidates
  std::vector<Candidate> candidates;
  std::vector<BlockWeight> block_weight_deltas_to(_p_graph.k());
  std::vector<BlockWeight> block_weight_deltas_from(_p_graph.k());

  START_TIMER("Find move candidates");
  for (const BlockID from : _p_graph.blocks()) {
    for (const auto &pq_element : _pq.elements(from)) {
      const NodeID &node = pq_element.id;
      const double &gain = (_nb_ctx.par_update_pq_gains ? pq_element.key : _tmp_gains[node]);

      if (block_overload(from) <= block_weight_deltas_from[from]) {
        break;
      }

      const BlockID to = _target_blocks[node];
      const auto bucket = _buckets.compute_bucket(gain);

      KASSERT(
          [&] {
            const auto max_gainer = _gain_calculator.compute_max_gainer(node, _p_ctx);

            if (gain != max_gainer.relative_gain()) {
              LOG_WARNING << "bad relative gain for node " << node << ": " << gain
                          << " != " << max_gainer.relative_gain();
              return false;
            }
            // Skip check: does not work when using the randomized gain calculator
            /*if (to != max_gainer.block) {
              LOG_WARNING << "bad target block for node " << node << ": " << to
                          << " != " << max_gainer.block;
              return false;
            }*/
            return true;
          }(),
          "inconsistent PQ gains",
          HEAVY
      );

      if (!_nb_ctx.par_partial_buckets || bucket < cutoff_buckets[from]) {
        Candidate candidate = {
            .id = _p_graph.local_to_global_node(node),
            .from = from,
            .to = to,
            .weight = _p_graph.node_weight(node),
            .gain = gain,
        };

        if (candidate.from == candidate.to) {
          [[maybe_unused]] const bool reassigned =
              assign_feasible_target_block(candidate, block_weight_deltas_to);
          KASSERT(
              reassigned,
              "could not find a feasible target block for node "
                  << candidate.id << ", weight " << candidate.weight << ", deltas: ["
                  << block_weight_deltas_to << "]"
                  << ", max block weights: " << _p_ctx.graph->max_block_weights
                  << ", block weights: "
                  << std::vector<BlockWeight>(
                         _p_graph.block_weights().begin(), _p_graph.block_weights().end()
                     )
          );
        }

        block_weight_deltas_to[candidate.to] += candidate.weight;
        block_weight_deltas_from[candidate.from] += candidate.weight;
        candidates.push_back(candidate);
      }
    }
  }
  STOP_TIMER();
  TIMER_BARRIER(_p_graph.communicator());

  // Compute total weight to each block
  START_TIMER("Allreduce weight to block");
  MPI_Allreduce(
      MPI_IN_PLACE,
      block_weight_deltas_to.data(),
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
          1.0 * block_underload(candidate.to) / block_weight_deltas_to[candidate.to];
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
  TIMER_BARRIER(_p_graph.communicator());

  if (balanced_moves || _nb_ctx.par_accept_imbalanced_moves) {
    for (const BlockID block : _p_graph.blocks()) {
      _p_graph.set_block_weight(
          block, _p_graph.block_weight(block) + actual_block_weight_deltas[block]
      );
    }

    candidates.resize(candidates.size() - num_rejected_candidates);

    START_TIMER("Perform moves");
    perform_moves(candidates, false);
    STOP_TIMER();
    TIMER_BARRIER(_p_graph.communicator());

    TIMED_SCOPE("Synchronize partition state after fast rebalance round") {
      struct Message {
        NodeID node;
        BlockID block;
      };

      mpi::graph::sparse_alltoall_interface_to_pe_custom_range<Message>(
          _p_graph.graph(),
          0,
          candidates.size(),
          [&](const NodeID i) -> NodeID { return _p_graph.global_to_local_node(candidates[i].id); },
          [&](NodeID) -> bool { return true; },
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

    TIMER_BARRIER(_p_graph.communicator());
    return true;
  }

  // Parallel rebalancing stalled
  return false;
}

bool NodeBalancer::is_sequential_balancing_enabled() const {
  return _stalled || _nb_ctx.enable_sequential_balancing;
}

bool NodeBalancer::is_parallel_balancing_enabled() const {
  return !_stalled && _nb_ctx.enable_parallel_balancing;
}

bool NodeBalancer::assign_feasible_target_block(
    Candidate &candidate, const std::vector<BlockWeight> &deltas
) const {
  do {
    ++candidate.to;
    if (candidate.to == _p_ctx.k) {
      candidate.to = 0;
    }
  } while (candidate.from != candidate.to &&
           block_underload(candidate.to) < candidate.weight + deltas[candidate.to]);

  return candidate.from != candidate.to;
}
} // namespace kaminpar::dist
