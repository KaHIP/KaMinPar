/*******************************************************************************
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 * @brief:  Distributed JET refiner due to:
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 ******************************************************************************/
#include "dkaminpar/refinement/jet_refiner.h"

#include <tbb/parallel_invoke.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/graphutils/synchronization.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/refinement/greedy_balancer.h"

#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

void JetRefiner::refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Jet");

  const NodeID min_size = p_ctx.k * _ctx.coarsening.contraction_limit;
  const NodeID cur_size = p_graph.n();
  const NodeID max_size = p_ctx.graph->n;
  const double min_c = _ctx.refinement.jet.min_c;
  const double max_c = _ctx.refinement.jet.max_c;
  const double c = [&] {
    if (_ctx.refinement.jet.interpolate_c) {
      return min_c + (max_c - min_c) * (cur_size - min_size) / (max_size - min_size);
    } else {
      if (cur_size <= 2 * min_size) {
        return min_c;
      } else {
        return max_c;
      }
    }
  }();
  DBG << "Setting c=" << c;

  START_TIMER("Allocation");
  NoinitVector<std::uint8_t> lock(p_graph.n());
  p_graph.pfor_nodes([&](const NodeID u) { lock[u] = 0; });

  GreedyBalancer balancer(_ctx);
  balancer.initialize(p_graph.graph());

  NoinitVector<BlockID> next_partition(p_graph.total_n());
  NoinitVector<BlockID> best_partition(p_graph.total_n());
  p_graph.pfor_all_nodes([&](const NodeID u) {
    const BlockID b = p_graph.block(u);
    next_partition[u] = b;
    best_partition[u] = b;
  });

  NoinitVector<EdgeWeight> gains(p_graph.total_n());

  EdgeWeight best_cut = metrics::edge_cut(p_graph);
  bool last_iteration_is_best = true;

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> rating_map_ets{[&] {
    return RatingMap<EdgeWeight, BlockID>(_ctx.partition.k);
  }};

  NoinitVector<NodeWeight> block_weight_deltas(p_graph.k());
  NoinitVector<NodeWeight> best_block_weights(p_graph.k());
  p_graph.pfor_blocks([&](const BlockID b) { block_weight_deltas[b] = 0; });
  p_graph.pfor_blocks([&](const BlockID b) { best_block_weights[b] = p_graph.block_weight(b); });
  STOP_TIMER();

  for (int i = 0; i < _ctx.refinement.jet.num_iterations; ++i) {
    const EdgeWeight initial_cut =
        _ctx.refinement.jet.use_abortion_threshold ? metrics::edge_cut(p_graph) : -1;

    TIMED_SCOPE("Find moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        const BlockID from = p_graph.block(u);
        const NodeWeight u_weight = p_graph.node_weight(u);

        if (lock[u]) {
          next_partition[u] = from;
          return;
        }

        BlockID max_gainer = from;
        EdgeWeight max_external_gain = std::numeric_limits<EdgeWeight>::min();
        EdgeWeight internal_degree = 0;

        auto compute_gain = [&](auto &map) {
          for (const auto [e, v] : p_graph.neighbors(u)) {
            const BlockID v_block = p_graph.block(v);
            const EdgeWeight weight = p_graph.edge_weight(e);
            if (from != v_block) {
              map[v_block] += weight;
            } else {
              internal_degree += weight;
            }
          }

          auto &rand = Random::instance();
          for (const auto [block, gain] : map.entries()) {
            if (gain > max_external_gain || (gain == max_external_gain && rand.random_bool())) {
              max_gainer = block;
              max_external_gain = gain;
            }
          }
          map.clear();
        };

        auto &rating_map = rating_map_ets.local();
        rating_map.update_upper_bound_size(std::min<NodeID>(p_ctx.k, p_graph.degree(u)));
        rating_map.run_with_map(compute_gain, compute_gain);

        // Filter internal nodes
        if (max_gainer == from) {
          next_partition[u] = from;
          return;
        }

        // Filter moves below negative gain threshold
        const EdgeWeight best_gain = max_external_gain - internal_degree;
        if (-best_gain >= std::floor(c * internal_degree)) {
          next_partition[u] = from;
          return;
        }

        // Accept move
        next_partition[u] = max_gainer;
        gains[u] = best_gain;
      });
    };

    TIMED_SCOPE("Exchange moves") {
      tbb::parallel_for<NodeID>(p_graph.n(), p_graph.total_n(), [&](const NodeID ghost) {
        next_partition[ghost] = p_graph.block(ghost);
      });

      struct Message {
        NodeID node;
        BlockID block;
        EdgeWeight gain;
      };

      mpi::graph::sparse_alltoall_interface_to_pe<Message>(
          p_graph.graph(),
          [&](const NodeID u) { return next_partition[u] != p_graph.block(u); },
          [&](const NodeID u) -> Message {
            return {
                .node = u,
                .block = next_partition[u],
                .gain = gains[u],
            };
          },
          [&](const auto &recv_buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
              const auto [their_lnode, block, gain] = recv_buffer[i];
              const auto gnode = static_cast<GlobalNodeID>(p_graph.offset_n(pe) + their_lnode);
              const NodeID lnode = p_graph.global_to_local_node(gnode);
              next_partition[lnode] = block;
              gains[lnode] = gain;
            });
          }
      );
    };

    TIMED_SCOPE("Filter moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        lock[u] = 0;

        const BlockID from = p_graph.block(u);
        const BlockID to = next_partition[u];
        if (from == to) {
          return;
        }

        const EdgeWeight gain_u = gains[u];
        EdgeWeight gain = 0;

        for (const auto &[e, v] : p_graph.neighbors(u)) {
          const EdgeWeight weight = p_graph.edge_weight(e);

          const bool v_before_u = [&, v = v] {
            if (next_partition[v] == p_graph.block(v)) {
              return false;
            }

            const EdgeWeight gain_v = gains[v];
            return gain_v > gain_u || (gain_v == gain_u && v < u);
          }();
          const BlockID block_v = v_before_u ? next_partition[v] : p_graph.block(v);

          if (to == block_v) {
            gain += weight;
          } else if (from == block_v) {
            gain -= weight;
          }
        }

        if (gain > 0) {
          lock[u] = 1;
        }
      });
    };

    TIMED_SCOPE("Execute moves") {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (lock[u]) {
          const BlockID from = p_graph.block(u);
          const BlockID to = next_partition[u];
          p_graph.set_block<false>(u, to);

          // @todo if k is small, use thread-local vectors
          __atomic_fetch_sub(&block_weight_deltas[from], p_graph.node_weight(u), __ATOMIC_RELAXED);
          __atomic_fetch_add(&block_weight_deltas[to], p_graph.node_weight(u), __ATOMIC_RELAXED);
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
          p_graph.graph(),
          [&](const NodeID u) { return lock[u]; }, // only look at nodes that were moved
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

      // Update block weights
      MPI_Allreduce(
          MPI_IN_PLACE,
          block_weight_deltas.data(),
          p_graph.k(),
          mpi::type::get<NodeWeight>(),
          MPI_SUM,
          p_graph.communicator()
      );
      p_graph.pfor_blocks([&](const BlockID b) {
        p_graph.set_block_weight(b, p_graph.block_weight(b) + block_weight_deltas[b]);
        block_weight_deltas[b] = 0;
      });

      KASSERT(graph::debug::validate_partition(p_graph), "", assert::normal);
    };

    TIMED_SCOPE("Rebalance") {
      balancer.refine(p_graph, p_ctx);
    };

    TIMED_SCOPE("Update best partition") {
      const EdgeWeight current_cut = metrics::edge_cut(p_graph);
      if (current_cut <= best_cut) {
        tbb::parallel_invoke(
            [&] {
              p_graph.pfor_all_nodes([&](const NodeID u) { best_partition[u] = p_graph.block(u); });
            },
            [&] {
              p_graph.pfor_blocks([&](const BlockID b) {
                best_block_weights[b] = p_graph.block_weight(b);
              });
            }
        );
        best_cut = current_cut;
        last_iteration_is_best = true;
      } else {
        last_iteration_is_best = false;
      }
    };

    if (_ctx.refinement.jet.use_abortion_threshold) {
      const EdgeWeight final_cut = metrics::edge_cut(p_graph);
      const double improvement = 1.0 * (initial_cut - final_cut) / initial_cut;
      if (1.0 - improvement > _ctx.refinement.jet.abortion_threshold) {
        break;
      }
    }
  }

  TIMED_SCOPE("Rollback") {
    if (!last_iteration_is_best) {
      tbb::parallel_invoke(
          [&] {
            p_graph.pfor_all_nodes([&](const NodeID u) {
              p_graph.set_block<false>(u, best_partition[u]);
            });
          },
          [&] {
            p_graph.pfor_blocks([&](const BlockID b) {
              p_graph.set_block_weight(b, best_block_weights[b]);
            });
          }
      );
    }
  };

  KASSERT(graph::debug::validate_partition(p_graph), "", assert::normal);
}
} // namespace kaminpar::dist

