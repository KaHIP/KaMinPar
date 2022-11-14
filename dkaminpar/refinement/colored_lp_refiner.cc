/***********************************************************************************************************************
 * @file:   colored_lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds determined by a graph coloring.
 **********************************************************************************************************************/
#include "dkaminpar/refinement/colored_lp_refiner.h"

#include <kassert/kassert.hpp>

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

#include "common/parallel/algorithm.h"
#include "common/timer.h"

namespace kaminpar::dist {
ColoredLPRefiner::ColoredLPRefiner(const Context& ctx) : _input_ctx(ctx) {}

void ColoredLPRefiner::initialize(const DistributedGraph& graph) {
    SCOPED_TIMER("Initialize colorized label propagation refiner");

    const auto    coloring   = compute_node_coloring_sequentially(graph, _input_ctx.refinement.lp.num_chunks);
    const ColorID num_colors = *std::max_element(coloring.begin(), coloring.end());

    TIMED_SCOPE("Allocation") {
        _color_sorted_nodes.resize(graph.n());
        _color_sizes.resize(num_colors + 1);
        tbb::parallel_for<std::size_t>(0, _color_sorted_nodes.size(), [&](const std::size_t i) {
            _color_sorted_nodes[i] = 0;
        });
        tbb::parallel_for<std::size_t>(0, _color_sizes.size(), [&](const std::size_t i) { _color_sizes[i] = 0; });
    };

    TIMED_SCOPE("Count color sizes") {
        graph.pfor_nodes([&](const NodeID u) {
            const ColorID c = coloring[u];
            KASSERT(c < num_colors);
            __atomic_fetch_add(&_color_sizes[c], 1, __ATOMIC_RELAXED);
        });
        parallel::prefix_sum(_color_sizes.begin(), _color_sizes.end(), _color_sizes.begin());
    };

    TIMED_SCOPE("Sort nodes") {
        graph.pfor_nodes([&](const NodeID u) {
            const ColorID     c = coloring[u];
            const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
            KASSERT(i < _color_sorted_nodes.size());
            _color_sorted_nodes[i] = u;
        });
    };

    KASSERT(_color_sizes.front() == 0u);
    KASSERT(_color_sizes.back() == graph.n());
}

void ColoredLPRefiner::refine(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    _p_ctx   = &p_ctx;
    _p_graph = &p_graph;

    for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
        const NodeID from = _color_sizes[c];
        const NodeID to   = _color_sizes[c + 1];

        for (NodeID seq_u = from; seq_u < to; ++seq_u) {
            const NodeID u = _color_sorted_nodes[seq_u];
            handle_node(u);
        }
    }
}

bool ColoredLPRefiner::perform_moves(
    const NodeID seq_from, const NodeID seq_to, const std::vector<BlockWeight>& residual_block_weights,
    const std::vector<EdgeWeight>& total_gains_to_block
) {
    struct Move {
        Move(const NodeID u, const BlockID from) : u(u), from(from) {}
        NodeID  u;
        BlockID from;
    };

    // perform probabilistic moves, but keep track of moves in case we need to roll back
    std::vector<parallel::Atomic<NodeWeight>>      expected_moved_weight(_p_ctx->k);
    scalable_vector<parallel::Atomic<BlockWeight>> block_weight_deltas(_p_ctx->k);
    tbb::concurrent_vector<Move>                   moves;
    _p_graph->pfor_nodes_range(from, to, [&](const auto& r) {
        auto& rand = Random::instance();

        for (NodeID u = r.begin(); u < r.end(); ++u) {
            // only iterate over nodes that changed block
            if (_next_partition[u] == _p_graph->block(u) || _next_partition[u] == kInvalidBlockID) {
                continue;
            }

            // compute move probability
            const BlockID b = _next_partition[u];
            const double  gain_prob =
                _lp_ctx.ignore_probabilities
                     ? 1.0
                     : ((total_gains_to_block[b] == 0) ? 1.0 : 1.0 * _gains[u] / total_gains_to_block[b]);
            const double probability =
                _lp_ctx.ignore_probabilities
                    ? 1.0
                    : gain_prob * (static_cast<double>(residual_block_weights[b]) / _p_graph->node_weight(u));
            IFSTATS(_statistics.expected_gain += probability * _gains[u]);
            IFSTATS(expected_moved_weight[b] += probability * _p_graph->node_weight(u));

            // perform move with probability
            if (_lp_ctx.ignore_probabilities || rand.random_bool(probability)) {
                IFSTATS(_statistics.num_tentatively_moved_nodes++);

                const BlockID    from     = _p_graph->block(u);
                const BlockID    to       = _next_partition[u];
                const NodeWeight u_weight = _p_graph->node_weight(u);

                moves.emplace_back(u, from);
                block_weight_deltas[from].fetch_sub(u_weight, std::memory_order_relaxed);
                block_weight_deltas[to].fetch_add(u_weight, std::memory_order_relaxed);
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
    scalable_vector<BlockWeight> global_block_weight_deltas(_p_ctx->k);
    _p_graph->pfor_blocks([&](const BlockID b) { global_block_weight_deltas[b] = block_weight_deltas[b]; });
    MPI_Allreduce(
        MPI_IN_PLACE, global_block_weight_deltas.data(), asserting_cast<int>(_p_ctx->k), mpi::type::get<BlockWeight>(),
        MPI_SUM, _p_graph->communicator()
    );

    // check for balance violations
    parallel::Atomic<std::uint8_t> feasible = 1;
    if (!_lp_ctx.ignore_probabilities) {
        _p_graph->pfor_blocks([&](const BlockID b) {
            if (_p_graph->block_weight(b) + global_block_weight_deltas[b] > max_cluster_weight(b)
                && global_block_weight_deltas[b] > 0) {
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
                const auto& move        = *it;
                _next_partition[move.u] = _p_graph->block(move.u);
                _p_graph->set_block<false>(move.u, move.from);

                IFSTATS(_statistics.rollback_gain += _gains[move.u]);
            }
        });
    } else { // otherwise, update block weights
        _p_graph->pfor_blocks([&](const BlockID b) {
            _p_graph->set_block_weight(b, _p_graph->block_weight(b) + global_block_weight_deltas[b]);
        });
    }

    // update block weights used by LP
    _p_graph->pfor_blocks([&](const BlockID b) { _block_weights[b] = _p_graph->block_weight(b); });

    // check that feasible is the same on all PEs
    if constexpr (kDebug) {
        int feasible_nonatomic = feasible;
        int root_feasible      = mpi::bcast(feasible_nonatomic, 0, _p_graph->communicator());
        KASSERT(root_feasible == feasible_nonatomic);
    }

    return feasible;
}

void ColoredLPRefiner::synchronize_state(const ColorID c) {
    struct MoveMessage {
        NodeID  local_node;
        BlockID new_block;
    };

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    mpi::graph::sparse_alltoall_interface_to_pe_custom_range<MoveMessage>(
        *_graph, seq_from, seq_to,

        // Map sequence index to node
        [&](const NodeID seq_u) { return _color_sorted_nodes[seq_u]; },

        // We set _next_partition[] to kInvalidBlockID for nodes that were moved during perform_moves()
        [&](const NodeID u) -> bool { return _next_partition[u] == kInvalidBlockID; },

        // Send move to each ghost node adjacent to u
        [&](const NodeID u) -> MoveMessage {
            // perform_moves() marks nodes that were moved locally by setting _next_partition[] to kInvalidBlockID
            // here, we revert this mark
            _next_partition[u] = _p_graph->block(u);

            return {.local_node = u, .new_block = _p_graph->block(u)};
        },

        // Move ghost nodes
        [&](const auto recv_buffer, const PEID pe) {
            tbb::parallel_for(static_cast<std::size_t>(0), recv_buffer.size(), [&](const std::size_t i) {
                const auto [local_node_on_pe, new_block] = recv_buffer[i];
                const auto   global_node = static_cast<GlobalNodeID>(_p_graph->offset_n(pe) + local_node_on_pe);
                const NodeID local_node  = _p_graph->global_to_local_node(global_node);
                KASSERT(new_block != _p_graph->block(local_node)); // otherwise, we should not have gotten this message

                _p_graph->set_block<false>(local_node, new_block);
            });
        }
    );
}

void ColoredLPRefiner::handle_node(const NodeID u) {}
} // namespace kaminpar::dist
