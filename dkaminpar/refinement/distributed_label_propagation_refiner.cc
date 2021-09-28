#include "dkaminpar/refinement/distributed_label_propagation_refiner.h"

#include "dkaminpar/mpi_graph_utils.h"
#include "dkaminpar/mpi_utils.h"
#include "dkaminpar/utility/distributed_math.h"
#include "dkaminpar/utility/vector_ets.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/datastructure/rating_map.h"
#include "kaminpar/utility/random.h"

namespace dkaminpar {
constexpr static NodeID kNodeToDebug = 292;

void DistributedLabelPropagationRefiner::initialize(const DistributedGraph & /* graph */,
                                                    const PartitionContext &p_ctx) {
  _p_ctx = &p_ctx;
}

void DistributedLabelPropagationRefiner::refine(DistributedPartitionedGraph &p_graph) {
  _p_graph = &p_graph;
  Base::initialize(&p_graph.graph(), _p_ctx->k); // needs access to _p_graph

  for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
    for (std::size_t chunk = 0; chunk < _lp_ctx.num_chunks; ++chunk) {
      const auto [from, to] = math::compute_local_range<NodeID>(_p_graph->n(), _lp_ctx.num_chunks, chunk);
      process_chunk(from, to);
    }
  }
}

void DistributedLabelPropagationRefiner::process_chunk(const NodeID from, const NodeID to) {
  mpi::barrier(_graph->communicator());
  HEAVY_ASSERT(ASSERT_NEXT_PARTITION_STATE());

  // run label propagation
  DBG << "in_order label propagation " << from << ".." << to;
  this->perform_iteration(from, to);

  // accumulate total weight of nodes moved to each block
  DBG << "compute weight and gain to each block";
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
}

bool DistributedLabelPropagationRefiner::perform_moves(const NodeID from, const NodeID to,
                                                       const std::vector<BlockWeight> &residual_block_weights,
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

      // perform move with probability
      if (rand.random_bool(probability)) {
        moves.emplace_back(u, _p_graph->block(u));
        _p_graph->set_block(u, _next_partition[u]);

        // temporary mark that this node was actually moved
        // we will revert this during synchronization or on rollback
        _next_partition[u] = kInvalidBlockID;
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

  // revert moves if resulting partition is infeasible
  if (!feasible) {
    for (const auto &move : moves) {
      _next_partition[move.u] = _p_graph->block(move.u);
      _p_graph->set_block(move.u, move.from);
    }
  }

  return feasible;
}

void DistributedLabelPropagationRefiner::synchronize_state(const NodeID from, const NodeID to) {
  struct MoveMessage {
    GlobalNodeID global_node;
    BlockID new_block;
  };

  mpi::graph::sparse_alltoall_interface_node_range_filtered<MoveMessage, scalable_vector>(
      *_graph, from, to,

      // only for nodes that were moved -- we set _next_partition[] to kInvalidBlockID for nodes that were actually
      // moved during perform_moves
      [&](const NodeID u) -> bool {
        const bool was_moved = (_next_partition[u] == kInvalidBlockID);
        _next_partition[u] = _p_graph->block(u);
        return was_moved;
      },

      // send move to each ghost node adjacent to u
      [&](const NodeID u, const PEID /* pe */) -> MoveMessage {
        return {.global_node = _p_graph->local_to_global_node(u), .new_block = _p_graph->block(u)};
      },

      // move ghost nodes
      [&](const PEID /* p */, const auto &recv_buffer) {
        tbb::parallel_for(static_cast<std::size_t>(0), recv_buffer.size(), [&](const std::size_t i) {
          const auto [global_node, new_block] = recv_buffer[i];
          const NodeID local_node = _p_graph->global_to_local_node(global_node);
          ASSERT(new_block != _p_graph->block(local_node)); // otherwise, we should not have gotten this message

          _p_graph->set_block(local_node, new_block);
        });
      });
}

#ifdef KAMINPAR_ENABLE_HEAVY_ASSERTIONS
bool DistributedLabelPropagationRefiner::ASSERT_NEXT_PARTITION_STATE() {
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
} // namespace dkaminpar