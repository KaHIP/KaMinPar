/*******************************************************************************
 * Greedy balancing algorithm that moves clusters of nodes at a time.
 *
 * @file:   cluster_balancer.cc
 * @author: Daniel Seemaier
 * @date:   19.07.2023
 ******************************************************************************/
#include "kaminpar-dist/refinement/balancer/cluster_balancer.h"

#include <iomanip>
#include <sstream>

#include "kaminpar-mpi/binary_reduction_tree.h"
#include "kaminpar-mpi/sparse_alltoall.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/refinement/balancer/clusters.h"
#include "kaminpar-dist/refinement/balancer/reductions.h"

#define HEAVY assert::heavy

namespace kaminpar::dist {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

void ClusterBalancer::Statistics::reset() {
  num_rounds = 0;
  cluster_stats.clear();

  num_seq_rounds = 0;
  num_seq_cluster_moves = 0;
  num_seq_node_moves = 0;
  seq_imbalance_reduction = 0.0;
  seq_cut_increase = 0;

  num_par_rounds = 0;
  num_par_cluster_moves = 0;
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
  STATS << "    Number of clusters moved: " << num_seq_cluster_moves << " with "
        << num_seq_node_moves << " nodes = " << 1.0 * num_seq_node_moves / num_seq_cluster_moves
        << " nodes per cluster";
  STATS << "    Imbalance reduction:      " << seq_imbalance_reduction << " = "
        << 1.0 * seq_imbalance_reduction / num_seq_cluster_moves << " per cluster, "
        << 1.0 * seq_imbalance_reduction / num_seq_node_moves << " per node";
  STATS << "    Cut increase:             " << seq_cut_increase << " = "
        << 1.0 * seq_cut_increase / num_seq_cluster_moves << " per cluster, "
        << 1.0 * seq_cut_increase / num_seq_node_moves << " per node";
  STATS << "  Parallel rounds:";
  STATS << "    Number of clusters moved: " << num_par_cluster_moves << " with "
        << num_par_node_moves << " nodes = " << 1.0 * num_par_node_moves / num_par_cluster_moves
        << " nodes per cluster";
  STATS << "    Imbalance reduction:      " << par_imbalance_reduction << " = "
        << 1.0 * par_imbalance_reduction / num_par_cluster_moves << " per cluster, "
        << 1.0 * par_imbalance_reduction / num_par_node_moves << " per node";
  STATS << "    Cut increase:             " << par_cut_increase << " = "
        << 1.0 * par_cut_increase / num_par_cluster_moves << " per cluster, "
        << 1.0 * par_cut_increase / num_par_node_moves << " per node";
  STATS << "    # of dicing attempts:     " << num_par_dicing_attempts << " = "
        << 1.0 * num_par_dicing_attempts / num_par_rounds << " per round";
  STATS << "    # of balanced moves:      " << num_par_balanced_moves;
  STATS << "  Number of cluster rebuilds: " << cluster_stats.size();
  for (std::size_t i = 0; i < cluster_stats.size(); ++i) {
    const auto &stats = cluster_stats[i];
    STATS << "    [" << i << "] # of clusters: " << stats.cluster_count << " with "
          << stats.node_count << " nodes";
    STATS << "    [" << i << "]                ... min: " << stats.min_cluster_size
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
      _cb_ctx(ctx.refinement.cluster_balancer),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _pqs(_p_graph.n(), _p_graph.k()),
      _pq_weights(_p_graph.k()),
      _moved_marker(_p_graph.n()),
      _weight_buckets(
          _p_graph, _p_ctx, _cb_ctx.par_use_positive_gain_buckets, _cb_ctx.par_gain_bucket_factor
      ),
      _current_parallel_rebalance_fraction(_cb_ctx.par_initial_rebalance_fraction) {}

ClusterBalancer::~ClusterBalancer() {
  _factory.take_m_ctx(std::move(*this));
}

ClusterBalancer::operator ClusterBalancerMemoryContext() && {
  return {
      std::move(_clusters),
  };
}

void ClusterBalancer::initialize() {
  SCOPED_TIMER("Cluster Balancer Initialization");

  IFSTATS(_stats.reset());
  _stalled = false;
  rebuild_clusters();
}

void ClusterBalancer::rebuild_clusters() {
  SCOPED_TIMER("Initialize clusters and data structures");

  init_clusters();
  clear();
  for (const NodeID cluster : _clusters.clusters()) {
    if (!is_overloaded(_clusters.block(cluster))) {
      continue;
    }
    try_pq_insertion(cluster);
  }

  // Barrier for time measurement
  mpi::barrier(_p_graph.communicator());
}

void ClusterBalancer::init_clusters() {
  _clusters = build_clusters(
      get_cluster_strategy(),
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
        .min_cluster_size = min,
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
      HEAVY
  );
}

void ClusterBalancer::clear() {
  _pqs.clear();
  std::fill(_pq_weights.begin(), _pq_weights.end(), 0);
  _moved_marker.reset();
  _weight_buckets.clear();
}

void ClusterBalancer::try_pq_insertion(const NodeID cluster) {
  KASSERT(!_pqs.contains(cluster));
  KASSERT(!_moved_marker.get(cluster));

  const BlockID from_block = _clusters.block(cluster);
  const auto [relative_gain, to_block] = _clusters.find_max_relative_gain(cluster);

  // Add weight to the correct weight bucket
  _weight_buckets.add(from_block, _clusters.weight(cluster), relative_gain);

  // Add this move set to the PQ if:
  // @todo seq_full_pq is mandatory for now, otherwise updating the weight buckets does not work
  bool accept = _cb_ctx.seq_full_pq;
  KASSERT(accept, "implementation does not work with partial PQs", assert::always);

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
      _pq_weights[to_block] += _clusters.weight(cluster);
    }

    _pq_weights[from_block] += _clusters.weight(cluster);
    _pqs.push(from_block, cluster, relative_gain);
  }
}

void ClusterBalancer::try_pq_update(const NodeID cluster) {
  const BlockID from_block = _clusters.block(cluster);
  const auto [relative_gain, to_block] = _clusters.find_max_relative_gain(cluster);

  KASSERT(
      !_moved_marker.get(cluster),
      "cluster " << cluster << " was already moved and should not be updated"
  );
  KASSERT(_pqs.contains(cluster), "cluster " << cluster << " not contained in the PQ");
  KASSERT(relative_gain != std::numeric_limits<double>::min(), "illegal relative gain");
  KASSERT(relative_gain != std::numeric_limits<double>::max(), "illegal relative gain");

  _weight_buckets.remove(from_block, _clusters.weight(cluster), _pqs.key(from_block, cluster));
  _pqs.change_priority(from_block, cluster, relative_gain);
  _weight_buckets.add(from_block, _clusters.weight(cluster), relative_gain);
}

bool ClusterBalancer::refine() {
  SCOPED_TIMER("Cluster Balancer Refinement");

  KASSERT(
      debug::validate_partition(_p_graph),
      "input partition for the move cluster balancer is in an inconsistent state",
      HEAVY
  );
  KASSERT(dbg_validate_pq_weights(), "PQ weights are inaccurate after initialization", HEAVY);
  KASSERT(
      dbg_validate_bucket_weights(), "bucket weights are inaccurate after initialization", HEAVY
  );

  const double initial_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
  double prev_imbalance_distance = initial_imbalance_distance;

  // @todo too expensive
  // EdgeWeight prev_edge_cut = 0;
  // IFSTATS(prev_edge_cut = metrics::edge_cut(_p_graph));

  bool force_cluster_rebuild = false;

  for (int round = 0; prev_imbalance_distance > 0 && round < _cb_ctx.max_num_rounds; ++round) {
    IFSTATS(++_stats.num_rounds);
    DBG0 << "Starting round " << round;

    if (force_cluster_rebuild || (round > 0 && _cb_ctx.cluster_rebuild_interval > 0 &&
                                  (round % _cb_ctx.cluster_rebuild_interval) == 0)) {
      DBG0 << "  --> rebuild move cluster after every " << _cb_ctx.cluster_rebuild_interval;
      rebuild_clusters();
      force_cluster_rebuild = false;
    }

    if (use_sequential_rebalancing()) {
      perform_sequential_round();
      DBG0 << "  --> Round " << round << ": seq. balancing: " << prev_imbalance_distance << " --> "
           << metrics::imbalance_l1(_p_graph, _p_ctx);

      IF_STATS {
        _stats.seq_imbalance_reduction +=
            (prev_imbalance_distance - metrics::imbalance_l1(_p_graph, _p_ctx));
        // @todo too expensive
        //  _stats.seq_cut_increase += metrics::edge_cut(_p_graph) - prev_edge_cut;
        //  prev_edge_cut = metrics::edge_cut(_p_graph);
      }
    }

    if (use_parallel_rebalancing()) {
      const double imbalance_after_seq_balancing = metrics::imbalance_l1(_p_graph, _p_ctx);
      if ((prev_imbalance_distance - imbalance_after_seq_balancing) / prev_imbalance_distance <
          _cb_ctx.parallel_threshold) {
        perform_parallel_round();
        DBG0 << "  --> Round " << round << ": par. balancing: " << imbalance_after_seq_balancing
             << " --> " << metrics::imbalance_l1(_p_graph, _p_ctx);

        IFSTATS(
            _stats.par_imbalance_reduction +=
            (imbalance_after_seq_balancing - metrics::imbalance_l1(_p_graph, _p_ctx))
        );

        // @todo too expensive
        // IFSTATS(_stats.par_cut_increase += metrics::edge_cut(_p_graph) - prev_edge_cut);
      }
    }

    // Abort if we couldn't improve balance
    const double next_imbalance_distance = metrics::imbalance_l1(_p_graph, _p_ctx);
    if (next_imbalance_distance >= prev_imbalance_distance) {
      if ((!_cb_ctx.switch_to_sequential_after_stallmate ||
           !_cb_ctx.switch_to_singleton_after_stallmate) &&
          _stalled) {
        if (mpi::get_comm_rank(_p_graph.communicator()) == 0) {
          LOG_WARNING << "Stallmate: imbalance distance " << next_imbalance_distance
                      << " could not be improved in round " << round;
          LOG_WARNING << dbg_get_pq_state_str() << " /// " << dbg_get_partition_state_str();
        }

        break;
      }

      _stalled = true;
      force_cluster_rebuild = true;
    }
    prev_imbalance_distance = next_imbalance_distance;

    KASSERT(dbg_validate_pq_weights(), "PQ weights are inaccurate after round " << round, HEAVY);
    KASSERT(
        dbg_validate_bucket_weights(), "bucket weights are inaccurate after round " << round, HEAVY
    );
  }

  KASSERT(
      debug::validate_partition(_p_graph),
      "partition is in an inconsistent state after round " << round,
      HEAVY
  );
  IFSTATS(_stats.print());
  return prev_imbalance_distance > 0;
}

void ClusterBalancer::perform_parallel_round() {
  SCOPED_TIMER("Parallel round");

  IFSTATS(++_stats.num_par_rounds);

  constexpr static bool kUseBinaryReductionTree = false;
  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  auto buckets = [&] {
    if constexpr (kUseBinaryReductionTree) {
      return reduce_buckets_binary_tree(_weight_buckets, _p_graph);
    } else {
      return reduce_buckets_mpireduce(_weight_buckets, _p_graph);
    }
  }();

  // Determine cut-off buckets and broadcast them to all PEs
  START_TIMER("Compute cut-off buckets");
  const BlockID num_overloaded_blocks = count_overloaded_blocks();
  std::vector<int> cut_off_buckets(num_overloaded_blocks);
  std::vector<BlockID> to_overloaded_map(_p_graph.k());
  BlockID current_overloaded_block = 0;

  for (const BlockID block : _p_graph.blocks()) {
    BlockWeight current_weight = _p_graph.block_weight(block);

    // We use a "fake" max_weight that only becomes the actual maximum block weight after a few
    // rounds This slows down rebalancing, but gives nodes in high-gain buckets a better chance for
    // being moved
    const BlockWeight max_weight = _p_ctx.graph->max_block_weight(block) +
                                   (1.0 - _current_parallel_rebalance_fraction) *
                                       (current_weight - _p_ctx.graph->max_block_weight(block));

    if (current_weight > max_weight) {
      if (rank == 0) {
        int cut_off_bucket = 0;
        for (; cut_off_bucket < _weight_buckets.num_buckets() && current_weight > max_weight;
             ++cut_off_bucket) {
          KASSERT(
              current_overloaded_block * _weight_buckets.num_buckets() + cut_off_bucket <
              buckets.size()
          );
          current_weight -=
              buckets[current_overloaded_block * _weight_buckets.num_buckets() + cut_off_bucket];
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
  STOP_TIMER();

  IF_DBG0 {
    std::stringstream ss;
    ss << "\n";
    ss << dbg_get_partition_state_str() << "\n";

    for (const BlockID b : _p_graph.blocks()) {
      if (overload(b) > 0) {
        ss << "Block " << b << ": ";
        for (std::size_t bucket = 0; bucket < _weight_buckets.num_buckets(); ++bucket) {
          ss << buckets[bucket + to_overloaded_map[b] * _weight_buckets.num_buckets()] << " ";
        }
        ss << " -- cut-off: " << cut_off_buckets[to_overloaded_map[b]] << "\n";
      }
    }
    DBG << ss.str();
  }

  START_TIMER("Pick candidates");
  std::vector<MoveCandidate> candidates;
  std::vector<BlockWeight> block_weight_deltas(_p_graph.k());

  for (const NodeID cluster : _clusters.clusters()) {
    if (_moved_marker.get(cluster)) {
      continue;
    }

    if (const BlockID from = _clusters.block(cluster); is_overloaded(from)) {
      auto [gain, to] = _clusters.find_max_relative_gain(cluster);
      const auto bucket = _weight_buckets.compute_bucket(gain);

      if (bucket < cut_off_buckets[to_overloaded_map[from]]) {
        const NodeWeight weight = _clusters.weight(cluster);

        MoveCandidate candidate = {
            .owner = rank,
            .id = cluster,
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
       !balanced_moves && attempt < std::max<int>(1, _cb_ctx.par_num_dicing_attempts);
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
  STOP_TIMER();

  IFSTATS(_stats.num_par_balanced_moves += balanced_moves);

  if (balanced_moves || _cb_ctx.par_accept_imbalanced) {
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

      _stats.num_par_cluster_moves += global_candidates_count;
      _stats.num_par_node_moves += dbg_count_nodes_in_clusters(candidates);
    }
  }

  // Increase rebalance fraction for next round
  _current_parallel_rebalance_fraction =
      std::min(1.0, _current_parallel_rebalance_fraction + _cb_ctx.par_rebalance_fraction_increase);
}

void ClusterBalancer::perform_sequential_round() {
  SCOPED_TIMER("Sequential round");

  IFSTATS(++_stats.num_seq_rounds);

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  // Step 1: identify the best move cluster candidates globally
  START_TIMER("Pick candidates");
  auto candidates = pick_sequential_candidates();
  candidates = reduce_candidates(
      pick_sequential_candidates(), _cb_ctx.seq_num_nodes_per_block, _p_graph, _p_ctx
  );
  STOP_TIMER();

  // Step 2: let ROOT decide which candidates to pick
  START_TIMER("Select winners");
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
      HEAVY
  );

  // Step 3: broadcast winners
  const std::size_t num_candidates = mpi::bcast(candidates.size(), 0, _p_graph.communicator());
  candidates.resize(num_candidates);
  mpi::bcast(candidates.data(), num_candidates, 0, _p_graph.communicator());
  STOP_TIMER();

  // Step 4: apply changes
  perform_moves(candidates, true);

  IF_STATS {
    _stats.num_seq_cluster_moves += candidates.size();
    _stats.num_seq_node_moves += dbg_count_nodes_in_clusters(candidates);
  }

  // Barrier for time measurement
  mpi::barrier(_p_graph.communicator());
}

void ClusterBalancer::perform_moves(
    const std::vector<MoveCandidate> &candidates, const bool update_block_weights
) {
  SCOPED_TIMER("Perform moves");

  KASSERT(
      dbg_validate_cluster_conns(), "cluster conns are inconsistent before performing moves", HEAVY
  );
  KASSERT(
      dbg_validate_bucket_weights(),
      "bucket weights are inconsistent before performing moves",
      HEAVY
  );

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());
  const PEID size = mpi::get_comm_size(_p_graph.communicator());

  struct MoveMessage {
    NodeID node;
    BlockID to;
  };
  Marker<> created_message_for_pe(size);
  std::vector<std::vector<MoveMessage>> move_sendbufs(size);

  auto update_adjacent_cluster = [&](const NodeID cluster) {
    if (is_overloaded(_clusters.block(cluster)) && !_moved_marker.get(cluster)) {
      if (!_pqs.contains(cluster)) {
        try_pq_insertion(cluster);
      } else {
        try_pq_update(cluster);
      }
    }
  };

  // By marking these clusters as marked, we avoid unnecessary updates if they are adjacent to other
  // moved clusters
  for (const auto &candidate : candidates) {
    if (rank == candidate.owner) {
      _moved_marker.set(candidate.id);
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

  for (const auto &candidate : candidates) {
    if (rank == candidate.owner) {
      _clusters.move_cluster(candidate.id, candidate.from, candidate.to);

      // We cannot use candidate.gain here, since that might not be the gain that was recomputed
      // during candidate selection
      KASSERT(_pqs.contains(candidate.id));
      _weight_buckets.remove(
          candidate.from, candidate.weight, _pqs.key(candidate.from, candidate.id)
      );

      _pq_weights[candidate.from] -= candidate.weight;
      _pqs.remove(candidate.from, candidate.id);

      for (NodeID u : _clusters.nodes(candidate.id)) {
        // @todo set blocks before updating other data structures to avoid max gainer changes?
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
          // new blocks that have not been overloaded when the clusters where created
          // --> also ignore nodes that are not assigned to any clusters
          if (_clusters.contains(v)) {
            update_adjacent_cluster(_clusters.cluster_of(v));
          }
        }

        created_message_for_pe.reset();
      }
    }
  }

  KASSERT(
      dbg_validate_cluster_conns(),
      "cluster conns are inconsistent after moving local nodes / local clusters",
      HEAVY
  );
  KASSERT(
      dbg_validate_bucket_weights(),
      "bucket weights are inconsistent after moving local nodes",
      HEAVY
  );

  mpi::sparse_alltoall<MoveMessage>(
      move_sendbufs,
      [&](const auto recvbuf, const PEID owner) {
        for (const auto &[their_lnode, to] : recvbuf) {
          const NodeID lnode = _p_graph.map_foreign_node(their_lnode, owner);
          _clusters.move_ghost_node(lnode, _p_graph.block(lnode), to, [&](const NodeID cluster) {
            update_adjacent_cluster(cluster);
          });
          _p_graph.set_block<false>(lnode, to);
        }
      },
      _p_graph.communicator()
  );

  KASSERT(
      dbg_validate_cluster_conns(), "cluster conns are inconsistent after moving ghost nodes", HEAVY
  );
  KASSERT(
      dbg_validate_bucket_weights(),
      "bucket weights are inconsistent after moving ghost nodes",
      HEAVY
  );
  KASSERT(dbg_validate_pq_weights(), "PQ weights are inconsistent after performing moves", HEAVY);
}

std::vector<ClusterBalancer::MoveCandidate> ClusterBalancer::pick_sequential_candidates() {
  SCOPED_TIMER("Pick sequential candidates");

  const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

  std::vector<MoveCandidate> candidates;
  for (const BlockID from : _p_graph.blocks()) {
    if (!is_overloaded(from)) {
      continue;
    }

    const std::size_t start = candidates.size();

    for (NodeID num = 0; num < _cb_ctx.seq_num_nodes_per_block; ++num) {
      if (_pqs.empty(from)) {
        break;
      }

      const NodeID cluster = _pqs.peek_max_id(from);
      const double relative_gain = _pqs.peek_max_key(from);
      const NodeWeight weight = _clusters.weight(cluster);
      _pqs.pop_max(from);
      _weight_buckets.remove(from, weight, relative_gain);

      auto [actual_relative_gain, to] = _clusters.find_max_relative_gain(cluster);
      if (actual_relative_gain >= relative_gain) {
        KASSERT(!_moved_marker.get(cluster));

        candidates.push_back(MoveCandidate{
            .owner = rank,
            .id = cluster,
            .weight = weight,
            .gain = actual_relative_gain,
            .from = from,
            .to = to,
        });
      } else {
        _pqs.push(from, cluster, actual_relative_gain);
        _weight_buckets.add(from, weight, actual_relative_gain);
        --num;
      }
    }

    for (auto candidate = candidates.begin() + start; candidate != candidates.end(); ++candidate) {
      _pqs.push(from, candidate->id, candidate->gain);
      _weight_buckets.add(from, candidate->weight, candidate->gain);
    }
  }

  return candidates;
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
  switch (_cb_ctx.cluster_size_strategy) {
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

  return limit * _cb_ctx.cluster_size_multiplier;
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
  std::vector<BlockWeight> local_block_weights(_p_graph.k());
  for (const NodeID u : _p_graph.nodes()) {
    local_block_weights[_p_graph.block(u)] += _p_graph.node_weight(u);
  }

  for (const BlockID block : _p_graph.blocks()) {
    if (is_overloaded(block)) {
      if (_pq_weights[block] == 0) {
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
    const BlockID block = _p_graph.block(u);
    if (overload(block) > 0 && _clusters.contains(u) &&
        !_moved_marker.get(_clusters.cluster_of(u))) {
      expected_total_weights[block] += _p_graph.node_weight(u);
    }
  }

  std::vector<BlockWeight> expected_per_bucket_weights(
      _weight_buckets.num_buckets() * _p_graph.k()
  );
  for (const NodeID cluster : _clusters.clusters()) {
    const BlockID block = _clusters.block(cluster);
    if (overload(block) > 0 && !_moved_marker.get(cluster)) {
      KASSERT(_pqs.contains(cluster));
      const std::size_t bucket = _weight_buckets.compute_bucket(_pqs.key(block, cluster));
      expected_per_bucket_weights[_weight_buckets.num_buckets() * block + bucket] +=
          _clusters.weight(cluster);
    }
  }

  for (const BlockID block : _p_graph.blocks()) {
    if (overload(block) == 0) {
      continue;
    }

    BlockWeight actual_total_weight = 0;
    for (std::size_t bucket = 0; bucket < _weight_buckets.num_buckets(); ++bucket) {
      if (_weight_buckets.size(block, bucket) < 0) {
        LOG_ERROR << "Block " << block << " has negative weight in bucket " << bucket;
        return false;
      }
      if (_weight_buckets.size(block, bucket) !=
          expected_per_bucket_weights[_weight_buckets.num_buckets() * block + bucket]) {
        LOG_ERROR << "Block " << block << " has " << _weight_buckets.size(block, bucket)
                  << " weight in bucket " << bucket << ", but its expected weight is "
                  << expected_per_bucket_weights[_weight_buckets.num_buckets() * block + bucket];
        return false;
      }
      actual_total_weight += _weight_buckets.size(block, bucket);
    }
    if (expected_total_weights[block] != actual_total_weight) {
      LOG_ERROR << "Block " << block << " has " << actual_total_weight
                << " weight in its buckets, but its expected weight is "
                << expected_total_weights[block];
      return false;
    }
  }

  return true;
}

bool ClusterBalancer::dbg_validate_cluster_conns() const {
  for (const NodeID cluster : _clusters.clusters()) {
    // Cluster connections are always updated, so we can check them for moved clusters too
    if (/* !_moved_marker.get(cluster) && */ !_clusters.dbg_check_conns(cluster)) {
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
      num_nodes += _clusters.size(candidate.id);
    }
  }
  MPI_Allreduce(
      MPI_IN_PLACE, &num_nodes, 1, mpi::type::get<NodeID>(), MPI_SUM, _p_graph.communicator()
  );
  return num_nodes;
}

bool ClusterBalancer::use_sequential_rebalancing() const {
  return (_stalled && _cb_ctx.switch_to_sequential_after_stallmate) ||
         _cb_ctx.enable_sequential_balancing;
}

bool ClusterBalancer::use_parallel_rebalancing() const {
  return (!_stalled || !_cb_ctx.switch_to_sequential_after_stallmate) &&
         _cb_ctx.enable_parallel_balancing;
}

ClusterStrategy ClusterBalancer::get_cluster_strategy() const {
  if (_stalled && _cb_ctx.switch_to_singleton_after_stallmate) {
    return ClusterStrategy::SINGLETONS;
  }

  return _cb_ctx.cluster_strategy;
}
} // namespace kaminpar::dist
