/*******************************************************************************
* @file:   distributed_probabilistic_label_propagation_refiner.h
*
* @author: Daniel Seemaier
* @date:   30.09.21
* @brief:
******************************************************************************/
#include "dkaminpar/refinement/distributed_probabilistic_label_propagation_refiner.h"

#include "dkaminpar/mpi_graph_utils.h"
#include "dkaminpar/mpi_utils.h"
#include "dkaminpar/utility/distributed_math.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "dkaminpar/utility/vector_ets.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/utility/random.h"

namespace dkaminpar {
struct DistributedLabelPropagationRefinerConfig : public shm::LabelPropagationConfig {
  using RatingMap = shm::RatingMap<EdgeWeight, shm::FastResetArray<EdgeWeight>>;
  using Graph = DistributedGraph;
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = false;
};

class DistributedProbabilisticLabelPropagationRefinerImpl final
    : public shm::InOrderLabelPropagation<DistributedProbabilisticLabelPropagationRefinerImpl,
                                          DistributedLabelPropagationRefinerConfig> {
  SET_STATISTICS(true);
  SET_DEBUG(false);

  using Base = shm::InOrderLabelPropagation<DistributedProbabilisticLabelPropagationRefinerImpl,
                                            DistributedLabelPropagationRefinerConfig>;

  struct Statistics {
    int num_successful_moves; // global
    int num_rollbacks;        // global

    double max_balance_violation;   // global, only if rollback occurred
    double total_balance_violation; // global, only if rollback occurred

    // local, expectation value of probabilistic gain values
    shm::parallel::IntegralAtomicWrapper<EdgeWeight> expected_gain;
    // local, gain values of moves that were executed
    shm::parallel::IntegralAtomicWrapper<EdgeWeight> realized_gain;
    // local, gain values that were rollbacked
    shm::parallel::IntegralAtomicWrapper<EdgeWeight> rollback_gain;
    // local, actual change in edge cut
    shm::parallel::IntegralAtomicWrapper<EdgeWeight> actual_gain;

    void reset() {
      num_successful_moves = 0;
      num_rollbacks = 0;
      max_balance_violation = 0.0;
      total_balance_violation = 0.0;
      expected_gain = 0;
      realized_gain = 0;
      rollback_gain = 0;
      actual_gain = 0;
    }

    void print() {
      auto expected_gain_reduced = mpi::reduce(expected_gain, MPI_SUM);
      auto realized_gain_reduced = mpi::reduce(realized_gain, MPI_SUM);
      auto rollback_gain_reduced = mpi::reduce(rollback_gain, MPI_SUM);

      STATS << "DistributedProbabilisticLabelPropagationRefiner:";
      STATS << "- Iterations: " << num_successful_moves << " ok, " << num_rollbacks << " failed";
      STATS << "- Expected gain: " << expected_gain_reduced << " (total expectation value of move gains)";
      STATS << "- Realized gain: " << realized_gain_reduced << " (total value of realized move gains)";
      STATS << "- Rollback gain: " << rollback_gain_reduced << " (gain of moves affected by rollback)";
      STATS << "- Actual gain: " << actual_gain << " (actual change in edge cut)";
      STATS << "- Balance violations: " << total_balance_violation / num_rollbacks << " / " << max_balance_violation;
    }
  };

public:
  explicit DistributedProbabilisticLabelPropagationRefinerImpl(const Context &ctx)
      : Base{ctx.partition.local_n()},
        _lp_ctx{ctx.refinement.lp},
        _next_partition(ctx.partition.local_n()),
        _gains(ctx.partition.local_n()),
        _block_weights(ctx.partition.k) {}

  void initialize(const DistributedGraph & /* graph */, const PartitionContext &p_ctx) {
    _p_ctx = &p_ctx;
    IFSTATS(_statistics.reset());
  }

  void refine(DistributedPartitionedGraph &p_graph) {
    _p_graph = &p_graph;
    Base::initialize(&p_graph.graph(), _p_ctx->k); // needs access to _p_graph

    const auto cut_before = IFSTATS(metrics::edge_cut(*_p_graph));

    for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      NodeID num_moved_nodes = 0;
      for (std::size_t chunk = 0; chunk < _lp_ctx.num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_p_graph->n(), _lp_ctx.num_chunks, chunk);
        num_moved_nodes += process_chunk(from, to);
      }
      if (num_moved_nodes == 0) { break; }
    }

    IFSTATS(_statistics.actual_gain += cut_before - metrics::edge_cut(*_p_graph));
    IFSTATS(_statistics.print());
  }

private:
  NodeID process_chunk(const NodeID from, const NodeID to) {
    mpi::barrier(_graph->communicator());
    HEAVY_ASSERT(ASSERT_NEXT_PARTITION_STATE());

    DBG << "Running label propagation on node chunk [" << from << ".." << to << "]";

    // run label propagation
    const NodeID num_moved_nodes = perform_iteration(from, to);
    if (num_moved_nodes == 0) { return 0; } // nothing to do

    // accumulate total weight of nodes moved to each block
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
    for (const BlockID b : _p_graph->blocks()) {
      const EdgeWeight global_gain_to = mpi::allreduce(gain_to_block[b], MPI_SUM, _graph->communicator());
      residual_cluster_weights.push_back(max_cluster_weight(b) - _p_graph->block_weight(b));
      global_total_gains_to_block.push_back(global_gain_to);
    }

    // perform probabilistic moves
    for (std::size_t i = 0; i < _lp_ctx.num_move_attempts; ++i) {
      if (perform_moves(from, to, residual_cluster_weights, global_total_gains_to_block)) {
        synchronize_state(from, to);
        break;
      }
    }

    // _next_partition should be in a consistent state at this point
    HEAVY_ASSERT(ASSERT_NEXT_PARTITION_STATE());
    return num_moved_nodes;
  }

  bool perform_moves(const NodeID from, const NodeID to, const std::vector<BlockWeight> &residual_block_weights,
                     const std::vector<EdgeWeight> &total_gains_to_block) {
    mpi::barrier(_graph->communicator());

    struct Move {
      NodeID u;
      BlockID from;
    };

    // perform probabilistic moves, but keep track of moves in case we need to roll back
    tbb::concurrent_vector<Move> moves;
    _p_graph->pfor_nodes_range(from, to, [&](const auto &r) {
      auto &rand = shm::Randomize::instance();

      for (NodeID u = r.begin(); u < r.end(); ++u) {
        // only iterate over nodes that changed block
        if (_next_partition[u] == _p_graph->block(u)) { return; }

        // compute move probability
        const BlockID b = _next_partition[u];
        const double gain_prob = (total_gains_to_block[b] == 0) ? 1.0 : 1.0 * _gains[u] / total_gains_to_block[b];
        const double probability = gain_prob * (1.0 * residual_block_weights[b] / _p_graph->node_weight(u));
        IFSTATS(_statistics.expected_gain += probability * _gains[u]);

        // perform move with probability
        if (rand.random_bool(probability)) {
          moves.emplace_back(u, _p_graph->block(u));
          _p_graph->set_block(u, _next_partition[u]);

          // temporary mark that this node was actually moved
          // we will revert this during synchronization or on rollback
          _next_partition[u] = kInvalidBlockID;

          IFSTATS(_statistics.realized_gain += _gains[u]);
        }
      }
    });

    // compute global block weights after moves
    std::vector<BlockWeight> global_block_weights(_p_graph->k());
    mpi::allreduce(_p_graph->block_weights_copy().data(), global_block_weights.data(), static_cast<int>(_p_graph->k()),
                   MPI_SUM, _graph->communicator());

    // check for balance violations
    shm::parallel::IntegralAtomicWrapper<std::uint8_t> feasible = 1;
    _p_graph->pfor_blocks([&](const BlockID b) {
      if (global_block_weights[b] > max_cluster_weight(b)) { feasible = 0; }
    });

    // record statistics
    if constexpr (kStatistics) {
      if (!feasible) {
        _statistics.num_rollbacks += 1;
        for (const BlockID b : _p_graph->blocks()) {
          if (global_block_weights[b] > max_cluster_weight(b)) {
            const double imbalance = global_block_weights[b] / max_cluster_weight(b);
            _statistics.total_balance_violation += imbalance;
            _statistics.max_balance_violation = std::max(_statistics.max_balance_violation, imbalance);
          }
        }
      } else {
        _statistics.num_successful_moves += 1;
      }
    }

    // revert moves if resulting partition is infeasible
    if (!feasible) {
      for (const auto &move : moves) {
        _next_partition[move.u] = _p_graph->block(move.u);
        _p_graph->set_block(move.u, move.from);

        IFSTATS(_statistics.rollback_gain += _gains[move.u]);
      }
    }

    return feasible;
  }

  void synchronize_state(const NodeID from, const NodeID to) {
    struct MoveMessage {
      GlobalNodeID global_node;
      BlockID new_block;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<MoveMessage>(
        *_graph, from, to,

        // only for nodes that were moved -- we set _next_partition[] to kInvalidBlockID for nodes that were actually
        // moved during perform_moves
        [&](const NodeID u) -> bool {
          const bool was_moved = (_next_partition[u] == kInvalidBlockID);
          _next_partition[u] = _p_graph->block(u);
          return was_moved;
        },

        // send move to each ghost node adjacent to u
        [&](const NodeID u) -> MoveMessage {
          return {.global_node = _p_graph->local_to_global_node(u), .new_block = _p_graph->block(u)};
        },

        // move ghost nodes
        [&](const auto recv_buffer) {
          tbb::parallel_for(static_cast<std::size_t>(0), recv_buffer.size(), [&](const std::size_t i) {
            const auto [global_node, new_block] = recv_buffer[i];
            const NodeID local_node = _p_graph->global_to_local_node(global_node);
            ASSERT(new_block != _p_graph->block(local_node)); // otherwise, we should not have gotten this message

            _p_graph->set_block(local_node, new_block);
          });
        });
  }

public:
  //
  // Called from base class
  //

  void init_cluster(const NodeID u, const BlockID b) { _next_partition[u] = b; }

  [[nodiscard]] BlockID cluster(const NodeID u) const { return _next_partition[u]; }

  void move_node(const NodeID u, const BlockID b) { _next_partition[u] = b; }

  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) const { return _p_graph->block_weight(b); }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID b) const { return _block_weights[b]; }

  void init_cluster_weight(const BlockID b, const BlockWeight weight) { _block_weights[b] = weight; }

  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID b) const { return _p_ctx->max_block_weight(b); }

  [[nodiscard]] bool move_cluster_weight(const BlockID from, const BlockID to, const BlockWeight delta,
                                         const BlockWeight max_weight) {
    if (_block_weights[to] + delta <= max_weight) {
      _block_weights[to] += delta;
      _block_weights[from] -= delta;
      return true;
    }
    return false;
  }

  [[nodiscard]] bool accept_cluster(const ClusterSelectionState &state) {
    const bool accept = (state.current_gain > state.best_gain ||
                         (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
                        (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
                         state.current_cluster == state.initial_cluster);
    if (accept) { _gains[state.u] = state.current_gain; }
    return accept;
  }

  [[nodiscard]] bool activate_neighbor(const NodeID u) const { return u < _p_graph->n(); }

private:
#ifdef KAMINPAR_ENABLE_HEAVY_ASSERTIONS
  bool ASSERT_NEXT_PARTITION_STATE() {
    mpi::barrier(_p_graph->communicator());
    for (const NodeID u : _p_graph->nodes()) {
      if (_p_graph->block(u) != _next_partition[u]) {
        LOG_ERROR << "Invalid _next_partition[] state for node " << u << ": " << V(_p_graph->block(u))
                  << V(_next_partition[u]);
        return false;
      }
    }
    mpi::barrier(_p_graph->communicator());
    return true;
  }
#endif

  const LabelPropagationRefinementContext &_lp_ctx;

  DistributedPartitionedGraph *_p_graph{nullptr};
  const PartitionContext *_p_ctx{nullptr};

  scalable_vector<BlockID> _next_partition;
  scalable_vector<EdgeWeight> _gains;
  scalable_vector<shm::parallel::IntegralAtomicWrapper<BlockWeight>> _block_weights;

  Statistics _statistics;
};

/*
 * Public interface
 */

DistributedProbabilisticLabelPropagationRefiner::DistributedProbabilisticLabelPropagationRefiner(const Context &ctx)
    : _impl{std::make_unique<DistributedProbabilisticLabelPropagationRefinerImpl>(ctx)} {}

DistributedProbabilisticLabelPropagationRefiner::~DistributedProbabilisticLabelPropagationRefiner() = default;

void DistributedProbabilisticLabelPropagationRefiner::initialize(const DistributedGraph &graph,
                                                                 const PartitionContext &p_ctx) {
  _impl->initialize(graph, p_ctx);
}

void DistributedProbabilisticLabelPropagationRefiner::refine(DistributedPartitionedGraph &p_graph) {
  _impl->refine(p_graph);
}
} // namespace dkaminpar