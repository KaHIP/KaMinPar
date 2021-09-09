/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi_graph_utils.h"
#include "dkaminpar/mpi_utils.h"
#include "dkaminpar/refinement/distributed_refiner.h"
#include "dkaminpar/utility/distributed_math.h"
#include "dkaminpar/utility/vector_ets.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/datastructure/rating_map.h"
#include "kaminpar/utility/random.h"

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

namespace dkaminpar {
struct DistributedLabelPropagationRefinerConfig : public shm::LabelPropagationConfig {
  using RatingMap = shm::RatingMap<EdgeWeight, shm::FastResetArray<EdgeWeight>>;
  using Graph = DistributedGraph;
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kUseStrictWeightConstraint = false;
  static constexpr bool kReportEmptyClusters = false;
};

class DistributedLabelPropagationRefiner final
    : public shm::LabelPropagation<DistributedLabelPropagationRefiner, DistributedLabelPropagationRefinerConfig>,
      DistributedRefiner {
  using Base = shm::LabelPropagation<DistributedLabelPropagationRefiner, DistributedLabelPropagationRefinerConfig>;

  SET_DEBUG(true);

public:
  explicit DistributedLabelPropagationRefiner(const Context &ctx)
      : Base{ctx.partition.local_n(), ctx.partition.k},
        _lp_ctx{ctx.refinement.lp},
        _next_partition(ctx.partition.local_n()),
        _gains(ctx.partition.local_n()) {}

  void initialize(const DistributedGraph & /* graph */, const PartitionContext &p_ctx) final { _p_ctx = &p_ctx; }

  void refine(DistributedPartitionedGraph &p_graph) final {
    _p_graph = &p_graph;
    Base::initialize(&p_graph.graph()); // needs access to _p_graph

    for (std::size_t iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      for (std::size_t chunk = 0; chunk < _lp_ctx.num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_p_graph->n(), _lp_ctx.num_chunks, chunk);
        process_chunk(from, to);
      }
    }
  }

private:
  void process_chunk(const NodeID from, const NodeID to) {
    mpi::barrier(_graph->communicator());
    HEAVY_ASSERT(ASSERT_NEXT_PARTITION_STATE());

    // run label propagation
    DBG << "in_order label propagation " << from << ".." << to;
    this->in_order_iteration(from, to);

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

  void synchronize_state(const NodeID from, const NodeID to) {
    struct MoveMessage {
      GlobalNodeID global_node;
      BlockID new_block;
    };

    mpi::graph::sparse_alltoall_interface_node_range_filtered<MoveMessage, scalable_vector>(
        *_graph, from, to,

        // only for nodes that were moved -- we set _next_partition[] to kInvalidBlockID for nodes that were actually
        // moved during perform_moves
        [&](const NodeID u) -> bool { return _next_partition[u] == kInvalidBlockID; },

        // send move to each ghost node adjacent to u
        [&](const NodeID u, const PEID /* pe */) -> MoveMessage {
          _next_partition[u] = _p_graph->block(u); // revert temporary mark that u was moved
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

public:
  void reset_node_state(const NodeID u) { _next_partition[u] = _p_graph->block(u); }
  [[nodiscard]] BlockID cluster(const NodeID u) const { return _next_partition[u]; }
  void set_cluster(const NodeID u, const BlockID b) { _next_partition[u] = b; }
  [[nodiscard]] BlockID num_clusters() const { return _p_graph->k(); }
  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) const { return _p_graph->block_weight(b); }
  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID b) const { return _p_ctx->max_block_weight(b); }
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
};
} // namespace dkaminpar