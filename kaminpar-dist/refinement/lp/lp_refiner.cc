/*******************************************************************************
 * Distributed label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-dist/refinement/lp/lp_refiner.h"

#include <mpi.h>

#include "kaminpar-mpi/sparse_allreduce.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/metrics.h"

#include "kaminpar-shm/label_propagation.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/vector_ets.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {
struct LPRefinerConfig : public LabelPropagationConfig {
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, BlockID, FastResetArray<EdgeWeight>>;
  using Graph = DistributedGraph;
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;

  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = false;
  static constexpr bool kUseActualGain = true;

  static constexpr bool kUseActiveSetStrategy = false;
  static constexpr bool kUseLocalActiveSetStrategy = true;
};

class LPRefinerImpl final : public ChunkRandomdLabelPropagation<LPRefinerImpl, LPRefinerConfig> {
  SET_STATISTICS(false);
  SET_DEBUG(false);

  using Base = ChunkRandomdLabelPropagation<LPRefinerImpl, LPRefinerConfig>;
  using Config = LPRefinerConfig;

  struct Statistics {
    Statistics(MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) {}

    MPI_Comm comm;

    EdgeWeight cut_before = 0;
    EdgeWeight cut_after = 0;

    int num_successful_moves = 0; // global
    int num_rollbacks = 0;        // global

    parallel::Atomic<int> num_tentatively_moved_nodes = 0;
    parallel::Atomic<int> num_tentatively_rejected_nodes = 0;

    double max_balance_violation = 0.0;   // global, only if rollback occurred
    double total_balance_violation = 0.0; // global, only if rollback occurred

    // local, expectation value of probabilistic gain values
    parallel::Atomic<EdgeWeight> expected_gain = 0;
    // local, gain values of moves that were executed
    parallel::Atomic<EdgeWeight> realized_gain = 0;
    parallel::Atomic<EdgeWeight> rejected_gain = 0;
    // local, gain values that were rollbacked
    parallel::Atomic<EdgeWeight> rollback_gain = 0;
    // local, expected imbalance
    double expected_imbalance = 0;

    void reset() {
      num_successful_moves = 0;
      num_rollbacks = 0;
      max_balance_violation = 0.0;
      total_balance_violation = 0.0;
      expected_gain = 0;
      realized_gain = 0;
      rejected_gain = 0;
      rollback_gain = 0;
      expected_imbalance = 0;
      num_tentatively_moved_nodes = 0;
      num_tentatively_rejected_nodes = 0;
    }

    void print() {
      auto expected_gain_reduced = mpi::reduce_single<EdgeWeight>(expected_gain, MPI_SUM, 0, comm);
      auto realized_gain_reduced = mpi::reduce_single<EdgeWeight>(realized_gain, MPI_SUM, 0, comm);
      auto rejected_gain_reduced = mpi::reduce_single<EdgeWeight>(rejected_gain, MPI_SUM, 0, comm);
      auto rollback_gain_reduced = mpi::reduce_single<EdgeWeight>(rollback_gain, MPI_SUM, 0, comm);
      auto expected_imbalance_str = mpi::gather_statistics_str(expected_imbalance, comm);
      auto num_tentatively_moved_nodes_str =
          mpi::gather_statistics_str(num_tentatively_moved_nodes.load(), comm);
      auto num_tentatively_rejected_nodes_str =
          mpi::gather_statistics_str(num_tentatively_rejected_nodes.load(), comm);

      STATS << "DistributedProbabilisticLabelPropagationRefiner:";
      STATS << "- Iterations: " << num_successful_moves << " ok, " << num_rollbacks << " failed";
      STATS << "- Expected gain: " << expected_gain_reduced
            << " (total expectation value of move gains)";
      STATS << "- Realized gain: " << realized_gain_reduced
            << " (total value of realized move gains)";
      STATS << "- Rejected gain: " << rejected_gain_reduced;
      STATS << "- Rollback gain: " << rollback_gain_reduced
            << " (gain of moves affected by rollback)";
      STATS << "- Actual gain: " << cut_before - cut_after << " (from " << cut_before << " to "
            << cut_after << ")";
      STATS << "- Balance violations: " << total_balance_violation / num_rollbacks << " / "
            << max_balance_violation;
      STATS << "- Expected imbalance: [" << expected_imbalance_str << "]";
      STATS << "- Num tentatively moved nodes: [" << num_tentatively_moved_nodes_str << "]";
      STATS << "- Num tentatively rejected nodes: [" << num_tentatively_rejected_nodes_str << "]";
    }
  };

public:
  explicit LPRefinerImpl(const Context &ctx, const DistributedPartitionedGraph &p_graph)
      : _lp_ctx(ctx.refinement.lp),
        _par_ctx(ctx.parallel),
        _next_partition(p_graph.n()),
        _gains(p_graph.n()),
        _block_weights(p_graph.k()) {
    set_max_degree(_lp_ctx.active_high_degree_threshold);
    allocate(p_graph.total_n(), p_graph.n(), p_graph.k());
  }

  void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("LP Refinement");
    _p_graph = &p_graph;
    _p_ctx = &p_ctx;

    // no of local nodes might increase on some PEs
    START_TIMER("Allocation");
    if (_next_partition.size() < p_graph.n()) {
      _next_partition.resize(p_graph.n());
    }
    if (_gains.size() < p_graph.n()) {
      _gains.resize(p_graph.n());
    }
    allocate(p_graph.total_n(), p_graph.n(), _block_weights.size());
    STOP_TIMER();

    Base::initialize(&p_graph.graph(), _p_ctx->k); // needs access to _p_graph

    IFSTATS(_statistics = Statistics{_p_graph->communicator()});
    IFSTATS(_statistics.cut_before = metrics::edge_cut(*_p_graph));

    const auto num_chunks = _lp_ctx.chunks.compute(_par_ctx);

    for (int iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      GlobalNodeID num_moved_nodes = 0;
      for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_p_graph->n(), num_chunks, chunk);
        num_moved_nodes += process_chunk(from, to);
      }
      if (num_moved_nodes == 0) {
        break;
      }
    }

    IFSTATS(_statistics.cut_after = metrics::edge_cut(*_p_graph));
    IFSTATS(_statistics.print());
  }

private:
  GlobalNodeID process_chunk(const NodeID from, const NodeID to) {
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT(ASSERT_NEXT_PARTITION_STATE(), "", assert::heavy);
#endif

    DBG << "Running label propagation on node chunk [" << from << ".." << to << "]";

    // run label propagation
    START_TIMER("Label propagation");
    const NodeID num_moved_nodes = perform_iteration(from, to);
    const auto global_num_moved_nodes =
        mpi::allreduce<GlobalNodeID>(num_moved_nodes, MPI_SUM, _graph->communicator());
    STOP_TIMER();

    if (global_num_moved_nodes == 0) {
      return 0; // nothing to do
    }

    // accumulate total weight of nodes moved to each block
    START_TIMER("Gather weight and gain values");
    parallel::vector_ets<BlockWeight> weight_to_block_ets(_p_ctx->k);
    parallel::vector_ets<EdgeWeight> gain_to_block_ets(_p_ctx->k);

    _p_graph->pfor_nodes_range(from, to, [&](const auto r) {
      auto &weight_to_block = weight_to_block_ets.local();
      auto &gain_to_block = gain_to_block_ets.local();

      for (NodeID u = r.begin(); u < r.end(); ++u) {
        if (_p_graph->block(u) != _next_partition[u]) {
          weight_to_block[_next_partition[u]] += _p_graph->node_weight(u);
          gain_to_block[_next_partition[u]] += _gains[u];
        }
      }
    });

    const auto weight_to_block = weight_to_block_ets.combine(std::plus{});
    const auto gain_to_block = gain_to_block_ets.combine(std::plus{});

    // allreduce gain to block
    std::vector<BlockWeight> residual_cluster_weights;
    std::vector<EdgeWeight> global_total_gains_to_block;

    // gather statistics
    std::vector<EdgeWeight> global_gain_to(_p_ctx->k);
    if (!_lp_ctx.ignore_probabilities) {
      mpi::allreduce(
          gain_to_block.data(),
          global_gain_to.data(),
          static_cast<int>(_p_ctx->k),
          MPI_SUM,
          _graph->communicator()
      );
    }

    for (const BlockID b : _p_graph->blocks()) {
      residual_cluster_weights.push_back(max_cluster_weight(b) - _p_graph->block_weight(b));
      global_total_gains_to_block.push_back(global_gain_to[b]);
    }
    STOP_TIMER();

    // perform probabilistic moves
    START_TIMER("Perform moves");
    for (int i = 0; i < _lp_ctx.num_move_attempts; ++i) {
      if (perform_moves(from, to, residual_cluster_weights, global_total_gains_to_block)) {
        break;
      }
    }
    synchronize_state(from, to);
    _p_graph->pfor_nodes(from, to, [&](const NodeID u) {
      _next_partition[u] = _p_graph->block(u);
    });
    STOP_TIMER();

    // _next_partition should be in a consistent state at this point
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT(ASSERT_NEXT_PARTITION_STATE(), "", assert::heavy);
#endif

    return global_num_moved_nodes;
  }

  bool perform_moves(
      const NodeID from,
      const NodeID to,
      const std::vector<BlockWeight> &residual_block_weights,
      const std::vector<EdgeWeight> &total_gains_to_block
  ) {
    mpi::barrier(_graph->communicator());
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT(debug::validate_partition(*_p_graph), "", assert::heavy);
#endif

    struct Move {
      Move(const NodeID u, const BlockID from) : u(u), from(from) {}
      NodeID u;
      BlockID from;
    };

    // perform probabilistic moves, but keep track of moves in case we need to
    // roll back
    NoinitVector<BlockWeight> block_weight_deltas(_p_ctx->k);
    tbb::parallel_for<BlockID>(0, _p_ctx->k, [&](const BlockID b) { block_weight_deltas[b] = 0; });

    tbb::concurrent_vector<Move> moves;

    _p_graph->pfor_nodes_range(from, to, [&](const auto &r) {
      auto &rand = Random::instance();

      for (NodeID u = r.begin(); u < r.end(); ++u) {
        // only iterate over nodes that changed block
        if (_next_partition[u] == _p_graph->block(u) || _next_partition[u] == kInvalidBlockID) {
          continue;
        }

        // compute move probability
        const BlockID b = _next_partition[u];
        const double gain_prob =
            _lp_ctx.ignore_probabilities
                ? 1.0
                : ((total_gains_to_block[b] == 0) ? 1.0 : 1.0 * _gains[u] / total_gains_to_block[b]
                  );
        const double probability =
            _lp_ctx.ignore_probabilities
                ? 1.0
                : gain_prob *
                      (static_cast<double>(residual_block_weights[b]) / _p_graph->node_weight(u));
        IFSTATS(_statistics.expected_gain += probability * _gains[u]);

        // perform move with probability
        if (_lp_ctx.ignore_probabilities || rand.random_bool(probability)) {
          IFSTATS(_statistics.num_tentatively_moved_nodes++);

          const BlockID from = _p_graph->block(u);
          const BlockID to = _next_partition[u];
          const NodeWeight u_weight = _p_graph->node_weight(u);

          moves.emplace_back(u, from);
          __atomic_fetch_sub(&block_weight_deltas[from], u_weight, __ATOMIC_RELAXED);
          __atomic_fetch_add(&block_weight_deltas[to], u_weight, __ATOMIC_RELAXED);
          _p_graph->set_block<false>(u, to);

          // temporary mark that this node was actually moved
          // we will revert this during synchronization or on rollback
          _next_partition[u] = kInvalidBlockID;

          IFSTATS(_statistics.realized_gain += _gains[u]);
        } else {
          IFSTATS(_statistics.num_tentatively_rejected_nodes++);
          IFSTATS(_statistics.rejected_gain += _gains[u]);
        }
      }
    });

    // compute global block weights after moves
    mpi::inplace_sparse_allreduce(
        block_weight_deltas, _p_ctx->k, MPI_SUM, _p_graph->communicator()
    );

    // check for balance violations
    parallel::Atomic<std::uint8_t> feasible = 1;
    if (!_lp_ctx.ignore_probabilities) {
      _p_graph->pfor_blocks([&](const BlockID b) {
        if (_p_graph->block_weight(b) + block_weight_deltas[b] > max_cluster_weight(b) &&
            block_weight_deltas[b] > 0) {
          feasible = 0;
        }
      });
    }

    // record statistics
    if constexpr (kStatistics) {
      if (!feasible) {
        _statistics.num_rollbacks += 1;
      } else {
        _statistics.num_successful_moves += 1;
      }
    }

    // revert moves if resulting partition is infeasible
    if (!feasible) {
      tbb::parallel_for(moves.range(), [&](const auto r) {
        for (auto it = r.begin(); it != r.end(); ++it) {
          const auto &move = *it;
          _next_partition[move.u] = _p_graph->block(move.u);
          _p_graph->set_block<false>(move.u, move.from);

          IFSTATS(_statistics.rollback_gain += _gains[move.u]);
        }
      });
    } else { // otherwise, update block weights
      _p_graph->pfor_blocks([&](const BlockID b) {
        _p_graph->set_block_weight(b, _p_graph->block_weight(b) + block_weight_deltas[b]);
      });
    }

    // update block weights used by LP
    _p_graph->pfor_blocks([&](const BlockID b) { _block_weights[b] = _p_graph->block_weight(b); });

    return feasible;
  }

  void synchronize_state(const NodeID from, const NodeID to) {
    struct MoveMessage {
      NodeID local_node;
      BlockID new_block;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<MoveMessage>(
        *_graph,
        from,
        to,

        // we set _next_partition[] to kInvalidBlockID for nodes that were moved
        // during perform_moves()
        [&](const NodeID u) -> bool { return _next_partition[u] == kInvalidBlockID; },

        // send move to each ghost node adjacent to u
        [&](const NodeID u) -> MoveMessage {
          // perform_moves() marks nodes that were moved locally by setting
          // _next_partition[] to kInvalidBlockID here, we revert this mark
          _next_partition[u] = _p_graph->block(u);

          return {.local_node = u, .new_block = _p_graph->block(u)};
        },

        // move ghost nodes
        [&](const auto recv_buffer, const PEID pe) {
          tbb::parallel_for(
              static_cast<std::size_t>(0),
              recv_buffer.size(),
              [&](const std::size_t i) {
                const auto [local_node_on_pe, new_block] = recv_buffer[i];
                const auto global_node =
                    static_cast<GlobalNodeID>(_p_graph->offset_n(pe) + local_node_on_pe);
                const NodeID local_node = _p_graph->global_to_local_node(global_node);
                KASSERT(new_block != _p_graph->block(local_node)); // otherwise, we should not
                                                                   // have gotten this message

                _p_graph->set_block<false>(local_node, new_block);
              }
          );
        }
    );
  }

public:
  //
  // Called from base class
  //

  void init_cluster(const NodeID u, const BlockID b) {
    KASSERT(u < _next_partition.size());
    _next_partition[u] = b;
  }

  [[nodiscard]] BlockID initial_cluster(const NodeID u) {
    KASSERT(u < _p_graph->n());
    return _p_graph->block(u);
  }

  [[nodiscard]] BlockID cluster(const NodeID u) {
    KASSERT(u < _p_graph->total_n());
    return _p_graph->is_owned_node(u) ? _next_partition[u] : _p_graph->block(u);
  }

  void move_node(const NodeID u, const BlockID b) {
    KASSERT(u < _p_graph->n());
    _next_partition[u] = b;
  }

  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) {
    return _p_graph->block_weight(b);
  }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID b) {
    return _block_weights[b];
  }

  void init_cluster_weight(const BlockID b, const BlockWeight weight) {
    _block_weights[b] = weight;
  }

  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID b) {
    return _p_ctx->graph->max_block_weight(b);
  }

  [[nodiscard]] bool move_cluster_weight(
      const BlockID from, const BlockID to, const BlockWeight delta, const BlockWeight max_weight
  ) {
    if (_block_weights[to] + delta <= max_weight) {
      _block_weights[to] += delta;
      _block_weights[from] -= delta;
      return true;
    }
    return false;
  }

  [[nodiscard]] bool accept_cluster(const typename Base::ClusterSelectionState &state) {
    const bool accept =
        (state.current_gain > state.best_gain ||
         (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
        (state.current_cluster_weight + state.u_weight <
             max_cluster_weight(state.current_cluster) ||
         state.current_cluster == state.initial_cluster);
    if (accept) {
      _gains[state.u] = state.current_gain;
    }
    return accept;
  }

  [[nodiscard]] bool activate_neighbor(const NodeID u) {
    return u < _p_graph->n();
  }

private:
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  bool ASSERT_NEXT_PARTITION_STATE() {
    mpi::barrier(_p_graph->communicator());
    for (const NodeID u : _p_graph->nodes()) {
      if (_p_graph->block(u) != _next_partition[u]) {
        LOG_ERROR << "Invalid _next_partition[] state for node " << u << ": "
                  << V(_p_graph->block(u)) << V(_next_partition[u]);
        return false;
      }
    }
    mpi::barrier(_p_graph->communicator());
    return true;
  }
#endif

  const LabelPropagationRefinementContext &_lp_ctx;
  const ParallelContext &_par_ctx;

  DistributedPartitionedGraph *_p_graph = nullptr;
  const PartitionContext *_p_ctx = nullptr;

  scalable_vector<BlockID> _next_partition;
  scalable_vector<EdgeWeight> _gains;
  scalable_vector<parallel::Atomic<BlockWeight>> _block_weights;

  Statistics _statistics;
};

/*
 * Public interface
 */

LPRefinerFactory::LPRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
LPRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<LPRefiner>(_ctx, p_graph, p_ctx);
}

LPRefiner::LPRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _impl(std::make_unique<LPRefinerImpl>(ctx, p_graph)),
      _p_graph(p_graph),
      _p_ctx(p_ctx) {}

LPRefiner::~LPRefiner() = default;

void LPRefiner::initialize() {}

bool LPRefiner::refine() {
  _impl->refine(_p_graph, _p_ctx);
  return false;
}
} // namespace kaminpar::dist
