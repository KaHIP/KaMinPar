/*******************************************************************************
 * Greedy balancing algorithm that moves clusters of nodes at a time.
 *
 * @file:   move_set_balancer.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "dkaminpar/refinement/balancer/cluster_balancer.h"

#include <iomanip>
#include <sstream>

#include "dkaminpar/mpi/binary_reduction_tree.h"
#include "dkaminpar/mpi/sparse_alltoall.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/refinement/balancer/clusters.h"

namespace kaminpar::dist {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(true);

void ClusterBalancer::Statistics::reset() {
  num_rounds = 0;
  cluster_stats.clear();

  num_seq_rounds = 0;
  num_seq_set_moves = 0;
  num_seq_node_moves = 0;
  seq_imbalance_reduction = 0.0;
  seq_cut_increase = 0;

  num_par_rounds = 0;
  num_par_set_moves = 0;
  num_par_node_moves = 0;
  num_par_dicing_attempts = 0;
  num_par_balanced_moves = 0;
  par_imbalance_reduction = 0.0;
  par_cut_increase = 0;
}

void ClusterBalancer::Statistics::print() {
  STATS << "Cluster Balancer:";
  STATS << "  Number of rounds:       " << num_rounds << " (" << num_seq_rounds << " sequential, "
        << num_par_rounds << " parallel)";
  STATS << "  Sequential rounds:";
  STATS << "    Number of clusters moved: " << num_seq_set_moves << " with " << num_seq_node_moves
        << " nodes = " << 1.0 * num_seq_node_moves / num_seq_set_moves << " nodes per cluster";
  STATS << "    Imbalance reduction:      " << seq_imbalance_reduction << " = "
        << 1.0 * seq_imbalance_reduction / num_seq_set_moves << " per cluster, "
        << 1.0 * seq_imbalance_reduction / num_seq_node_moves << " per node";
  STATS << "    Cut reduction:            " << seq_cut_increase << " = "
        << 1.0 * seq_cut_increase / num_seq_set_moves << " per cluster, "
        << 1.0 * seq_cut_increase / num_seq_node_moves << " per node";
  STATS << "  Parallel rounds:";
  STATS << "    Number of clusters moved: " << num_par_set_moves << " with " << num_par_node_moves
        << " nodes = " << 1.0 * num_par_node_moves / num_par_set_moves << " nodes per cluster";
  STATS << "    Imbalance reduction:      " << par_imbalance_reduction << " = "
        << 1.0 * par_imbalance_reduction / num_par_set_moves << " per cluster, "
        << 1.0 * par_imbalance_reduction / num_par_node_moves << " per node";
  STATS << "    Cut reduction:            " << par_cut_increase << " = "
        << 1.0 * par_cut_increase / num_par_set_moves << " per cluster, "
        << 1.0 * par_cut_increase / num_par_node_moves << " per node";
  STATS << "    # of dicing attempts:     " << num_par_dicing_attempts << " = "
        << 1.0 * num_par_dicing_attempts / num_par_rounds << " per round";
  STATS << "    # of balanced moves:      " << num_par_balanced_moves;
  STATS << "  Number of cluster rebuilds: " << cluster_stats.size();
  for (std::size_t i = 0; i < cluster_stats.size(); ++i) {
    const auto &stats = cluster_stats[i];
    STATS << "    [" << i << "] # of clusters: " << stats.cluster_count << " with "
          << stats.node_count << " nodes";
    STATS << "    [" << i << "]                ... min: " << stats.min_set_size
          << ", avg: " << 1.0 * stats.node_count / stats.cluster_count
          << ", max: " << stats.max_cluster_size;
  }
}

struct ClusterBalancerMemoryContext {
  ClustersMemoryContext move_sets_m_ctx;
};

ClusterBalancerFactory::ClusterBalancerFactory(const Context &ctx)
    : _ctx(ctx),
      _m_ctx(std::make_unique<ClusterBalancerMemoryContext>()) {}

ClusterBalancerFactory::~ClusterBalancerFactory() = default;

std::unique_ptr<GlobalRefiner> ClusterBalancerFactory::create(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  return std::make_unique<ClusterBalancer>(*this, _ctx, p_graph, p_ctx, std::move(*_m_ctx));
}

void ClusterBalancerFactory::take_m_ctx(ClusterBalancerMemoryContext m_ctx) {
  *_m_ctx = std::move(m_ctx);
}

ClusterBalancer::ClusterBalancer(
    ClusterBalancerFactory &factory,
    const Context &ctx,
    DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx,
    ClusterBalancerMemoryContext m_ctx
)
    : _factory(factory),
      _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _pqs(_p_graph.n(), _p_graph.k()),
      _pq_weights(_p_graph.k()),
      _moved_marker(_p_graph.n()),
      _weight_buckets(_p_graph, _p_ctx) {}

ClusterBalancer::~ClusterBalancer() {
  _factory.take_m_ctx(std::move(*this));
}

ClusterBalancer::operator ClusterBalancerMemoryContext() && {
  return {
      std::move(_clusters),
  };
}

void ClusterBalancer::initialize() {
  IFSTATS(_stats.reset());
  rebuild_clusters();
}

void ClusterBalancer::rebuild_clusters() {
  init_clusters();
  clear();
  for (const NodeID set : _clusters.clusters()) {
    if (!is_overloaded(_clusters.block(set))) {
      continue;
    }
    try_pq_insertion(set);
  }
}

void ClusterBalancer::init_clusters() {
  _clusters = build_clusters(
      _ctx.refinement.cluster_balancer.cluster_strategy,
      _p_graph,
      _ctx,
      _p_ctx,
      compute_cluster_weight_limit(),
      std::move(_clusters)
  );

  IF_STATS {
    NodeID set_count = _clusters.num_clusters();
    MPI_Allreduce(
        MPI_IN_PLACE, &set_count, 1, mpi::type::get<NodeID>(), MPI_SUM, _p_graph.communicator()
    );

    NodeID node_count = std::accumulate(
        _clusters.clusters().begin(),
        _clusters.clusters().end(),
        0u,
        [&](const NodeID sum, const NodeID set) { return sum + _clusters.size(set); }
    );
    MPI_Allreduce(
        MPI_IN_PLACE, &node_count, 1, mpi::type::get<NodeID>(), MPI_SUM, _p_graph.communicator()
    );

    NodeID min = std::numeric_limits<NodeID>::max();
    NodeID max = std::numeric_limits<NodeID>::min();
    for (const NodeID set : _clusters.clusters()) {
      min = std::min<NodeID>(min, _clusters.size(set));
      max = std::max<NodeID>(max, _clusters.size(set));
    }
    MPI_Allreduce(
        MPI_IN_PLACE, &min, 1, mpi::type::get<NodeID>(), MPI_MIN, _p_graph.communicator()
    );
    MPI_Allreduce(
        MPI_IN_PLACE, &max, 1, mpi::type::get<NodeID>(), MPI_MAX, _p_graph.communicator()
    );

    _stats.cluster_stats.push_back(ClusterStatistics{
        .cluster_count = set_count,
        .node_count = node_count,
        .min_set_size = min,
        .max_cluster_size = max,
    });
  }

  KASSERT(
      [&] {
        for (const NodeID node : _p_graph.nodes()) {
          if (is_overloaded(_p_graph.block(node)) && _clusters.cluster_of(node) == kInvalidNodeID) {
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
}

void ClusterBalancer::clear() {
  _pqs.clear();
  std::fill(_pq_weights.begin(), _pq_weights.end(), 0);
  _moved_marker.reset();
  _weight_buckets.clear();
}

void ClusterBalancer::try_pq_insertion(const NodeID set) {
  KASSERT(!_pqs.contains(set));
  KASSERT(!_moved_marker.get(set));

  const BlockID from_block = _clusters.block(set);
  const auto [relative_gain, to_block] = _clusters.find_max_relative_gain(set);

  // Add weight to the correct weight bucket
  _weight_buckets.add(from_block, _clusters.weight(set), relative_gain);

  // Add this move set to the PQ if:
  bool accept = _ctx.refinement.cluster_balancer.seq_full_pq;
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
      _pq_weights[from_block] -= _clusters.weight(replaced_set);
      _pq_weights[to_block] += _clusters.weight(set);
    }

    _pq_weights[from_block] += _clusters.weight(set);
    _pqs.push(from_block, set, relative_gain);
  }
}

void ClusterBalancer::try_pq_update(const NodeID set) {
  const BlockID from_block = _clusters.block(set);
  const auto [relative_gain, to_block] = _clusters.find_max_relative_gain(set);

  KASSERT(!_moved_marker.get(set), "set " << set << " was already moved and should not be updated");
  KASSERT(_pqs.contains(set), "set " << set << " not contained in the PQ");
  KASSERT(relative_gain != std::numeric_limits<double>::min(), "illegal relative gain");
  KASSERT(relative_gain != std::numeric_limits<double>::max(), "illegal relative gain");

  _weight_buckets.remove(from_block, set, _pqs.key(from_block, set));
  _pqs.change_priority(from_block, set, relative_gain);
  _weight_buckets.add(from_block, set, relative_gain);
}

bool ClusterBalancer::refine() {
  KASSERT(
      graph::debug::validate_partition(_p_graph),
      "input partition for the move set balancer is in an inconsistent state",
      assert::heavy
  );
  KASSERT(
      dbg_validate_pq_weights(), "PQ weights are inaccurate after initialization", assert::heavy
  );
  KASSERT(
      dbg_validate_bucket_weights(),
      "bucket weights are inaccurate after initialization",
      assert::normal
  ); // @todo change to heavy

  const double initial_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
  double prev_imbalance_distance = initial_imbalance_distance;

  EdgeWeight prev_edge_cut = 0;
  IFSTATS(prev_edge_cut = metrics::edge_cut(_p_graph));

  for (int round = 0;
       prev_imbalance_distance > 0 && round < _ctx.refinement.cluster_balancer.max_num_rounds;
       ++round) {
    IFSTATS(++_stats.num_rounds);
    DBG0 << "Starting round " << round;

    if (round > 0 && _ctx.refinement.cluster_balancer.cluster_rebuild_interval > 0 &&
        (round % _ctx.refinement.cluster_balancer.cluster_rebuild_interval) == 0) {
      DBG0 << "  --> rebuild move sets after every "
           << _ctx.refinement.cluster_balancer.cluster_rebuild_interval;

      rebuild_clusters();
    }

    if (_ctx.refinement.cluster_balancer.enable_sequential_balancing) {
      perform_sequential_round();
      DBG0 << "  --> Round " << round << ": seq. balancing: " << prev_imbalance_distance << " --> "
           << metrics::imbalance_l1(_p_graph, _p_ctx);
      IFSTATS(
          _stats.seq_imbalance_reduction +=
          (prev_imbalance_distance - metrics::imbalance_l1(_p_graph, _p_ctx))
      );
      IFSTATS(_stats.seq_cut_increase += metrics::edge_cut(_p_graph) - prev_edge_cut);
      IFSTATS(prev_edge_cut = metrics::edge_cut(_p_graph));
    }

    if (_ctx.refinement.cluster_balancer.enable_parallel_balancing) {
      const double imbalance_after_seq_balancing = metrics::imbalance_l1(_p_graph, _p_ctx);
      if ((prev_imbalance_distance - imbalance_after_seq_balancing) / prev_imbalance_distance <
          _ctx.refinement.cluster_balancer.parallel_threshold) {
        perform_parallel_round();
        DBG0 << "  --> Round " << round << ": par. balancing: " << imbalance_after_seq_balancing
             << " --> " << metrics::imbalance_l1(_p_graph, _p_ctx);
        IFSTATS(
            _stats.par_imbalance_reduction +=
            (imbalance_after_seq_balancing - metrics::imbalance_l1(_p_graph, _p_ctx))
        );
        IFSTATS(_stats.par_cut_increase += metrics::edge_cut(_p_graph) - prev_edge_cut);
      }
    }

    // Abort if we couldn't improve balance
    const double next_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
    if (next_imbalance_distance >= prev_imbalance_distance) {
      if (mpi::get_comm_rank(_p_graph.communicator()) == 0) {
        LOG_WARNING << "Stallmate: imbalance distance " << next_imbalance_distance
                    << " could not be improved in round " << round;
        LOG_WARNING << dbg_get_pq_state_str() << " /// " << dbg_get_partition_state_str();
      }
      break;
    } else {
      prev_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
    }

    KASSERT(
        dbg_validate_pq_weights(), "PQ weights are inaccurate after round " << round, assert::heavy
    );
    KASSERT(
        dbg_validate_bucket_weights(),
        "bucket weights are inaccurate after round " << round,
        assert::normal
    ); // @todo change to heavy
  }

  KASSERT(
      graph::debug::validate_partition(_p_graph),
      "partition is in an inconsistent state after round " << round,
      assert::heavy
  );
  IFSTATS(_stats.print());
  return prev_imbalance_distance > 0;
}

void ClusterBalancer::perform_parallel_round() {
  IFSTATS(++_stats.num_par_rounds);

  constexpr static bool kUseBinaryReductionTree = true;
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  auto buckets = [&] {
    if constexpr (kUseBinaryReductionTree) {
      auto compactified = _weight_buckets.compactify();
      StaticArray<GlobalNodeWeight> empty(compactified.size());

      return mpi::perform_binary_reduction(
          std::move(compactified),
          std::move(empty),
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

  IF_DBG0 {
    DBG << "---";
    DBG << dbg_get_partition_state_str();
    std::stringstream ss;
    ss << "\n";

    for (const BlockID b : _p_graph.blocks()) {
      if (overload(b) > 0) {
        ss << "Block " << b << ": ";
        for (std::size_t bucket = 0; bucket < Buckets::kNumBuckets; ++bucket) {
          ss << buckets[bucket + to_overloaded_map[b] * Buckets::kNumBuckets] << " ";
        }
        ss << " -- cut-off: " << cut_off_buckets[to_overloaded_map[b]] << "\n";
      }
    }
    DBG << ss.str();
    DBG << "---";
  }

  std::vector<MoveCandidate> candidates;
  std::vector<BlockWeight> block_weight_deltas(_p_graph.k());

  for (const NodeID set : _clusters.clusters()) {
    const BlockID from = _clusters.block(set);

    if (is_overloaded(from)) {
      auto [gain, to] = _clusters.find_max_relative_gain(set);
      const auto bucket = Buckets::compute_bucket(gain);

      if (bucket < cut_off_buckets[to_overloaded_map[from]]) {
        const NodeWeight weight = _clusters.weight(set);

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
  std::size_t num_rejected_candidates;
  std::vector<BlockWeight> actual_block_weight_deltas;
  bool balanced_moves = false;

  for (int attempt = 0;
       !balanced_moves &&
       attempt < std::max<int>(1, _ctx.refinement.cluster_balancer.par_num_dicing_attempts);
       ++attempt) {
    IFSTATS(++_stats.num_par_dicing_attempts);

    num_rejected_candidates = 0;
    actual_block_weight_deltas.clear();
    actual_block_weight_deltas.resize(_p_graph.k());

    for (std::size_t i = 0; i < candidates.size() - num_rejected_candidates; ++i) {
      const auto &candidate = candidates[i];
      const double probability = 1.0 * underload(candidate.to) / block_weight_deltas[candidate.to];
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
      if (overload(block) == 0 && underload(block) < actual_block_weight_deltas[block]) {
        balanced_moves = false;
        break;
      }
    }
  }

  IFSTATS(_stats.num_par_balanced_moves += balanced_moves);

  if (balanced_moves || _ctx.refinement.cluster_balancer.par_accept_imbalanced) {
    for (const BlockID block : _p_graph.blocks()) {
      _p_graph.set_block_weight(
          block, _p_graph.block_weight(block) + actual_block_weight_deltas[block]
      );
    }

    candidates.resize(candidates.size() - num_rejected_candidates);
    perform_moves(candidates, false);

    IF_STATS {
      std::size_t global_candidates_count = candidates.size();
      MPI_Allreduce(
          MPI_IN_PLACE,
          &global_candidates_count,
          1,
          mpi::type::get<std::size_t>(),
          MPI_SUM,
          _p_graph.communicator()
      );

      _stats.num_par_set_moves += global_candidates_count;
      _stats.num_par_node_moves += dbg_count_nodes_in_clusters(candidates);
    }
  }
}

void ClusterBalancer::perform_sequential_round() {
  IFSTATS(++_stats.num_seq_rounds);

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  // Step 1: identify the best move set candidates globally
  auto candidates = pick_sequential_candidates();

  candidates = reduce_sequential_candidates(pick_sequential_candidates());

  // Step 2: let ROOT decide which candidates to pick
  std::vector<BlockWeight> tmp_block_weight_deltas(_p_graph.k());

  for (const auto &candidate : candidates) { // empty on non-ROOT
    if (candidate.from == candidate.to) {
      continue;
    }
    tmp_block_weight_deltas[candidate.from] -= candidate.weight;
    tmp_block_weight_deltas[candidate.to] += candidate.weight;
  }

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

  KASSERT(
      [&] {
        for (const BlockID block : _p_graph.blocks()) {
          if (_p_graph.block_weight(block) <= _p_ctx.graph->max_block_weight(block) &&
              _p_graph.block_weight(block) + tmp_block_weight_deltas[block] >
                  _p_ctx.graph->max_block_weight(block)) {
            LOG_WARNING << "block " << block
                        << " was not overloaded before picking move candidates, but after adding "
                        << tmp_block_weight_deltas[block] << " weight to it, it is overloaded";
            return false;
          }
        }
        return true;
      }(),
      "picked candidates overload at least one block",
      assert::heavy
  );

  // Step 3: broadcast winners
  const std::size_t num_candidates = mpi::bcast(candidates.size(), 0, _p_graph.communicator());
  candidates.resize(num_candidates);
  mpi::bcast(candidates.data(), num_candidates, 0, _p_graph.communicator());

  // Step 4: apply changes
  perform_moves(candidates, true);

  IFSTATS(_stats.num_seq_set_moves += candidates.size());
  IFSTATS(_stats.num_seq_node_moves += dbg_count_nodes_in_clusters(candidates));
}

void ClusterBalancer::perform_moves(
    const std::vector<MoveCandidate> &candidates, const bool update_block_weights
) {
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
      _clusters.move_cluster(candidate.set, candidate.from, candidate.to);

      // We track moved sets to exclude them from further rounds
      _moved_marker.set(candidate.set);

      if (_pqs.contains(candidate.set)) {
        _pq_weights[candidate.from] -= candidate.weight;
        _pqs.remove(candidate.from, candidate.set);
      }
      _weight_buckets.remove(candidate.from, candidate.set, candidate.gain);

      for (NodeID u : _clusters.nodes(candidate.set)) {
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

          if (const NodeID set = _clusters.cluster_of(v);
              is_overloaded(_p_graph.block(v)) && _clusters.contains(v) && set != candidate.set &&
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
          _clusters.move_ghost_node(lnode, _p_graph.block(lnode), to);
          _p_graph.set_block<false>(lnode, to);
        }
      },
      _p_graph.communicator()
  );
}

std::vector<ClusterBalancer::MoveCandidate> ClusterBalancer::pick_sequential_candidates() {
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  std::vector<MoveCandidate> candidates;
  for (const BlockID from : _p_graph.blocks()) {
    if (!is_overloaded(from)) {
      continue;
    }

    const std::size_t start = candidates.size();

    for (NodeID num = 0; num < _ctx.refinement.cluster_balancer.seq_num_nodes_per_block; ++num) {
      if (_pqs.empty(from)) {
        break;
      }

      const NodeID set = _pqs.peek_max_id(from);
      const double relative_gain = _pqs.peek_max_key(from);
      const NodeWeight weight = _clusters.weight(set);
      _pqs.pop_max(from);

      auto [actual_relative_gain, to] = _clusters.find_max_relative_gain(set);
      if (actual_relative_gain >= relative_gain) {
        candidates.push_back(MoveCandidate{
            .owner = rank,
            .set = set,
            .weight = weight,
            .gain = actual_relative_gain,
            .from = from,
            .to = to,
        });
      } else {
        _pqs.push(from, set, actual_relative_gain);
        --num;
      }
    }

    for (auto candidate = candidates.begin() + start; candidate != candidates.end(); ++candidate) {
      _pqs.push(from, candidate->set, candidate->gain);
    }
  }

  return candidates;
}

std::vector<ClusterBalancer::MoveCandidate>
ClusterBalancer::reduce_sequential_candidates(std::vector<MoveCandidate> sendbuf) {
  return mpi::perform_binary_reduction(
      std::move(sendbuf),
      std::vector<MoveCandidate>{},
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
        std::vector<MoveCandidate> winners;

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
                      _ctx.refinement.cluster_balancer.seq_num_nodes_per_block) {
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

          winners.insert(
              winners.end(), candidates.begin(), candidates.begin() + num_accepted_candidates
          );

          idx_lhs = idx_lhs_end;
          idx_rhs = idx_rhs_end;
        }

        // Keep remaining nodes
        auto add_remaining_candidates = [&](const auto &vec, std::size_t i) {
          for (; i < vec.size(); ++i) {
            const BlockID from = vec[i].from;
            const BlockID to = vec[i].to;
            const NodeWeight weight = vec[i].weight;

            if (from == to || _p_graph.block_weight(to) + block_weight_deltas[to] + weight <=
                                  _p_ctx.graph->max_block_weight(to)) {
              winners.push_back(vec[i]);
              if (from != to) {
                block_weight_deltas[to] += weight;
              }
            }
          }
        };

        add_remaining_candidates(lhs, idx_lhs);
        add_remaining_candidates(rhs, idx_rhs);

        return winners;
      },
      _p_graph.communicator()
  );
}

BlockWeight ClusterBalancer::overload(const BlockID block) const {
  static_assert(std::is_signed_v<BlockWeight>);
  return std::max<BlockWeight>(
      0, _p_graph.block_weight(block) - _p_ctx.graph->max_block_weight(block)
  );
}

BlockWeight ClusterBalancer::underload(const BlockID block) const {
  static_assert(std::is_signed_v<BlockWeight>);
  return std::max<BlockWeight>(
      0, _p_ctx.graph->max_block_weight(block) - _p_graph.block_weight(block)
  );
}

bool ClusterBalancer::is_overloaded(const BlockID block) const {
  return overload(block) > 0;
}

BlockID ClusterBalancer::count_overloaded_blocks() const {
  return metrics::num_imbalanced_blocks(_p_graph, _p_ctx);
}

bool ClusterBalancer::assign_feasible_target_block(
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

NodeWeight ClusterBalancer::compute_cluster_weight_limit() const {
  NodeWeight limit = 0;
  switch (_ctx.refinement.cluster_balancer.cluster_size_strategy) {
  case ClusterSizeStrategy::ZERO:
    limit = 0;
    break;

  case ClusterSizeStrategy::ONE:
    limit = 1;
    break;

  case ClusterSizeStrategy::MAX_OVERLOAD:
    for (const BlockID block : _p_graph.blocks()) {
      limit = std::max<NodeWeight>(limit, overload(block));
    }
    break;

  case ClusterSizeStrategy::AVG_OVERLOAD:
    for (const BlockID block : _p_graph.blocks()) {
      limit += overload(block);
    }
    limit /= metrics::num_imbalanced_blocks(_p_graph, _p_ctx);
    break;

  case ClusterSizeStrategy::MIN_OVERLOAD:
    for (const BlockID block : _p_graph.blocks()) {
      limit = std::min<NodeWeight>(limit, overload(block));
    }
    break;
  }

  return limit * _ctx.refinement.cluster_balancer.cluster_size_multiplier;
}

std::string ClusterBalancer::dbg_get_partition_state_str() const {
  std::stringstream ss;
  ss << "Overloaded blocks: ";
  for (const BlockID block : _p_graph.blocks()) {
    if (is_overloaded(block)) {
      ss << "[" << std::setw(3) << block << ":" << std::setw(5) << overload(block) << "] ";
    }
  }
  return ss.str();
}

std::string ClusterBalancer::dbg_get_pq_state_str() const {
  std::stringstream ss;
  ss << "PQ size: " << _pqs.size() << " -> ";
  for (const BlockID block : _p_graph.blocks()) {
    ss << "[" << std::setw(3) << block << ":" << std::setw(5) << _pqs.size(block) << "] ";
  }
  return ss.str();
}

bool ClusterBalancer::dbg_validate_pq_weights() const {
  if (!_ctx.refinement.cluster_balancer.enable_sequential_balancing) {
    return true;
  }

  std::vector<BlockWeight> local_block_weights(_p_graph.k());
  for (const NodeID u : _p_graph.nodes()) {
    local_block_weights[_p_graph.block(u)] += _p_graph.node_weight(u);
  }

  for (const BlockID block : _p_graph.blocks()) {
    if (is_overloaded(block)) {
      if (_pq_weights[block] == 0) {
        /*
        LOG_WARNING << "Block " << block
                    << " is overloaded, but its PQ is empty -- this might happen if parallel "
                       "rebalance overloaded some block";
        */
        continue;
      }

      const BlockWeight expected_min_weight = overload(block);
      if (expected_min_weight > _pq_weights[block] &&
          _pq_weights[block] < local_block_weights[block]) {
        LOG_ERROR << "Block " << block << " has overload " << overload(block)
                  << ", but there is only " << _pq_weights[block] << " weight in its PQ";
        return false;
      }
    }
  }

  std::vector<BlockWeight> actual_weights(_p_graph.k());
  for (const NodeID node : _p_graph.nodes()) {
    if (_clusters.contains(node) && _pqs.contains(_clusters.cluster_of(node))) {
      actual_weights[_p_graph.block(node)] += _p_graph.node_weight(node);
    }
  }
  for (const BlockID block : _p_graph.blocks()) {
    if (actual_weights[block] != _pq_weights[block]) {
      LOG_ERROR << "Block " << block << " has " << actual_weights[block]
                << " weight in its PQ, but its PQ weight is " << _pq_weights[block];
      return false;
    }
  }

  return true;
}

bool ClusterBalancer::dbg_validate_bucket_weights() const {
  std::vector<BlockWeight> expected_total_weights(_p_graph.k());
  for (const NodeID u : _p_graph.nodes()) {
    const BlockID b = _p_graph.block(u);
    if (overload(b) > 0) {
      expected_total_weights[b] += _p_graph.node_weight(u);
    }
  }

  std::vector<BlockWeight> expected_per_bucket_weights(Buckets::kNumBuckets * _p_graph.k());
  for (const NodeID cluster : _clusters.clusters()) {
    const BlockID b = _clusters.block(cluster);
    if (overload(b) > 0) {
      const std::size_t bucket =
          Buckets::compute_bucket(_clusters.find_max_relative_gain(cluster).first);
      expected_per_bucket_weights[Buckets::kNumBuckets * b + bucket] += _clusters.weight(cluster);
    }
  }

  for (const BlockID b : _p_graph.blocks()) {
    BlockWeight actual_total_weight = 0;
    for (std::size_t bucket = 0; bucket < Buckets::kNumBuckets; ++bucket) {
      if (_weight_buckets.size(b, bucket) < 0) {
        LOG_ERROR << "Block " << b << " has negative weight " << _weight_buckets.size(b, bucket)
                  << " in bucket " << bucket;
        return false;
      }
      if (_weight_buckets.size(b, bucket) !=
          expected_per_bucket_weights[Buckets::kNumBuckets * b + bucket]) {
        LOG_ERROR << "Block " << b << " has " << _weight_buckets.size(b, bucket)
                  << " weight in bucket " << bucket << ", but its expected weight is "
                  << expected_per_bucket_weights[Buckets::kNumBuckets * b + bucket];
        return false;
      }
      actual_total_weight += _weight_buckets.size(b, bucket);
    }
    if (expected_total_weights[b] != actual_total_weight) {
      LOG_ERROR << "Block " << b << " has " << actual_total_weight
                << " weight in its buckets, but its expected weight is "
                << expected_total_weights[b];
      return false;
    }
  }

  return true;
}

NodeID ClusterBalancer::dbg_count_nodes_in_clusters(const std::vector<MoveCandidate> &candidates
) const {
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());
  NodeID num_nodes = 0;
  for (const auto &candidate : candidates) {
    if (candidate.owner == rank) {
      num_nodes += _clusters.size(candidate.set);
    }
  }
  MPI_Allreduce(
      MPI_IN_PLACE, &num_nodes, 1, mpi::type::get<NodeID>(), MPI_SUM, _p_graph.communicator()
  );
  return num_nodes;
}
} // namespace kaminpar::dist
