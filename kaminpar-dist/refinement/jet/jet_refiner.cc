/*******************************************************************************
 * Distributed JET refiner due to: "Jet: Multilevel Graph Partitioning on GPUs"
 * by Gilbert et al.
 *
 * @file:   jet_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#include "kaminpar-dist/refinement/jet/jet_refiner.h"

#include <tbb/parallel_invoke.h>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/graphutils/synchronization.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/refinement/gain_calculator.h"
#include "kaminpar-dist/refinement/snapshooter.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
JetRefinerFactory::JetRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
JetRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<JetRefiner>(_ctx, p_graph, p_ctx);
}

JetRefiner::JetRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _jet_ctx(ctx.refinement.jet),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _balancer(factory::create_refiner(_ctx, _ctx.refinement.jet.balancing_algorithm)
                    ->create(_p_graph, _p_ctx)),
      _locks(p_ctx.graph->n) {}

void JetRefiner::initialize() {}

bool JetRefiner::refine() {
  SCOPED_TIMER("Jet");

  START_TIMER("Allocation");
  BestPartitionSnapshooter snapshooter(_p_graph, _p_ctx);
  GainCalculator gain_calculator(_p_graph);

  NoinitVector<std::pair<EdgeWeight, BlockID>> gains_and_targets(_p_graph.total_n());
  _p_graph.pfor_all_nodes([&](const NodeID u) { gains_and_targets[u] = {0, _p_graph.block(u)}; });

  NoinitVector<NodeWeight> block_weight_deltas(_p_graph.k());
  _p_graph.pfor_blocks([&](const BlockID b) { block_weight_deltas[b] = 0; });
  STOP_TIMER();

  const double penalty_factor = compute_penalty_factor();

  const int max_num_fruitless_iterations = (_ctx.refinement.jet.num_fruitless_iterations == 0)
                                               ? std::numeric_limits<int>::max()
                                               : _ctx.refinement.jet.num_fruitless_iterations;
  const int max_num_iterations = (_ctx.refinement.jet.num_iterations == 0)
                                     ? std::numeric_limits<int>::max()
                                     : _ctx.refinement.jet.num_iterations;

  int cur_fruitless_iteration = 0;
  int cur_iteration = 0;

  do {
    const EdgeWeight initial_cut = metrics::edge_cut(_p_graph);

    TIMED_SCOPE("Find moves") {
      _p_graph.pfor_nodes([&](const NodeID u) {
        const BlockID b_u = _p_graph.block(u);
        const NodeWeight w_u = _p_graph.node_weight(u);

        if (_locks[u]) {
          gains_and_targets[u] = {0, b_u};
          return;
        }

        const auto max_gainer = gain_calculator.compute_max_gainer(u);

        // Filter internal nodes
        if (max_gainer.block == b_u ||
            max_gainer.ext_degree <= (1.0 - penalty_factor) * max_gainer.int_degree) {
          gains_and_targets[u] = {0, b_u};
        } else {
          gains_and_targets[u] = {max_gainer.absolute_gain(), max_gainer.block};
        }
      });
    };

    TIMED_SCOPE("Exchange moves") {
      tbb::parallel_for<NodeID>(_p_graph.n(), _p_graph.total_n(), [&](const NodeID ghost) {
        gains_and_targets[ghost] = {0, _p_graph.block(ghost)};
      });

      struct Message {
        NodeID node;
        EdgeWeight gain;
        BlockID block;
      };

      mpi::graph::sparse_alltoall_interface_to_pe<Message>(
          _p_graph.graph(),
          [&](const NodeID u) { return gains_and_targets[u].second != _p_graph.block(u); },
          [&](const NodeID u) -> Message {
            return {
                .node = u,
                .gain = gains_and_targets[u].first,
                .block = gains_and_targets[u].second,
            };
          },
          [&](const auto &recv_buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
              const auto [their_lnode, gain, block] = recv_buffer[i];
              const NodeID lnode = _p_graph.map_foreign_node(their_lnode, pe);
              gains_and_targets[lnode] = {gain, block};
            });
          }
      );
    };

    TIMED_SCOPE("Filter moves") {
      _p_graph.pfor_nodes([&](const NodeID u) {
        _locks[u] = 0;

        const BlockID from = _p_graph.block(u);
        const BlockID to = gains_and_targets[u].second;
        if (from == to) {
          return;
        }

        const EdgeWeight gain_u = gains_and_targets[u].first;
        EdgeWeight gain = 0;

        for (const auto &[e, v] : _p_graph.neighbors(u)) {
          const EdgeWeight weight = _p_graph.edge_weight(e);

          const bool v_before_u = [&, v = v] {
            if (gains_and_targets[v].second == _p_graph.block(v)) {
              return false;
            }

            const EdgeWeight gain_v = gains_and_targets[v].first;
            return gain_v > gain_u || (gain_v == gain_u && v < u);
          }();
          const BlockID b_v = v_before_u ? gains_and_targets[v].first : _p_graph.block(v);

          if (to == b_v) {
            gain += weight;
          } else if (from == b_v) {
            gain -= weight;
          }
        }

        if (gain > 0) {
          _locks[u] = 1;
        }
      });
    };

    TIMED_SCOPE("Execute moves") {
      _p_graph.pfor_nodes([&](const NodeID u) {
        if (_locks[u]) {
          const BlockID from = _p_graph.block(u);
          const BlockID to = gains_and_targets[u].second;
          _p_graph.set_block<false>(u, to);

          __atomic_fetch_sub(&block_weight_deltas[from], _p_graph.node_weight(u), __ATOMIC_RELAXED);
          __atomic_fetch_add(&block_weight_deltas[to], _p_graph.node_weight(u), __ATOMIC_RELAXED);
        }
      });
    };

    TIMED_SCOPE("Synchronize blocks") {
      // Synchronize block IDs of ghost nodes
      struct Message {
        NodeID node;
        BlockID block;
      };

      mpi::graph::sparse_alltoall_interface_to_pe<Message>(
          _p_graph.graph(),
          [&](const NodeID u) { return _locks[u]; }, // only look at nodes that were moved
          [&](const NodeID u) -> Message { return {.node = u, .block = _p_graph.block(u)}; },
          [&](const auto &recv_buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
              const auto [their_lnode, block] = recv_buffer[i];
              const NodeID lnode = _p_graph.map_foreign_node(their_lnode, pe);
              _p_graph.set_block<false>(lnode, block);
            });
          }
      );

      // Update block weights
      MPI_Allreduce(
          MPI_IN_PLACE,
          block_weight_deltas.data(),
          _p_graph.k(),
          mpi::type::get<NodeWeight>(),
          MPI_SUM,
          _p_graph.communicator()
      );
      _p_graph.pfor_blocks([&](const BlockID b) {
        _p_graph.set_block_weight(b, _p_graph.block_weight(b) + block_weight_deltas[b]);
        block_weight_deltas[b] = 0;
      });

      KASSERT(graph::debug::validate_partition(_p_graph), "", assert::normal);
    };

    TIMED_SCOPE("Rebalance") {
      _balancer->initialize();
      _balancer->refine();
    };

    TIMED_SCOPE("Update best partition") {
      snapshooter.update();
    };

    ++cur_iteration;
    ++cur_fruitless_iteration;

    const EdgeWeight final_cut = metrics::edge_cut(_p_graph);
    const double improvement = 1.0 * (initial_cut - final_cut) / initial_cut;
    if (improvement >= 1.0 - _ctx.refinement.jet.fruitless_threshold) {
      DBG << "Improved cut from " << initial_cut << " to " << final_cut
          << ": resetting number of fruitless iterations (threshold: "
          << _ctx.refinement.jet.fruitless_threshold << ")";
      cur_fruitless_iteration = 0;
    } else {
      DBG << "Fruitless edge cut change from " << initial_cut << " to " << final_cut
          << " (threshold: " << _ctx.refinement.jet.fruitless_threshold
          << "): incrementing fruitless iterations counter  to " << cur_fruitless_iteration;
    }
  } while (cur_iteration < max_num_iterations &&
           cur_fruitless_iteration < max_num_fruitless_iterations);

  TIMED_SCOPE("Rollback") {
    snapshooter.rollback();
  };

  KASSERT(graph::debug::validate_partition(_p_graph), "", assert::normal);
  return false;
}

void JetRefiner::reset_locks() {
  _p_graph.pfor_nodes([&](const NodeID u) { _locks[u] = 0; });
}

double JetRefiner::compute_penalty_factor() const {
  return (_p_graph.n() <= 2 * _p_ctx.k * _ctx.coarsening.contraction_limit)
             ? _jet_ctx.coarse_penalty_factor
             : _jet_ctx.fine_penalty_factor;
}
} // namespace kaminpar::dist
