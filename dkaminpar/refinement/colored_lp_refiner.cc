/***********************************************************************************************************************
 * @file:   colored_lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 * @brief:  Distributed label propagation refiner that moves nodes in rounds determined by a graph coloring.
 **********************************************************************************************************************/
#include "dkaminpar/refinement/colored_lp_refiner.h"

#include <kassert/kassert.hpp>
#include <tbb/enumerable_thread_specific.h>

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/graph_communication.h"

#include "common/datastructures/rating_map.h"
#include "common/parallel/algorithm.h"
#include "common/parallel/vector_ets.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

ColoredLPRefiner::ColoredLPRefiner(const Context& ctx) : _input_ctx(ctx) {}

void ColoredLPRefiner::initialize(const DistributedGraph& graph) {
    SCOPED_TIMER("Color label propagation refinement", "Initialization");

    const auto    coloring   = compute_node_coloring_sequentially(graph, _input_ctx.refinement.lp.num_chunks);
    const ColorID num_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    STATS << "Number of colors: " << num_colors;

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
    SCOPED_TIMER("Colored label propagation refinement", "Refinement");
    _p_ctx   = &p_ctx;
    _p_graph = &p_graph;

    TIMED_SCOPE("Allocation") {
        KASSERT(_next_partition.size() == _gains.size());
        if (_next_partition.size() < _p_graph->n()) {
            _next_partition.resize(_p_graph->n());
            _gains.resize(_p_graph->n());

            // Attribute first touch running time to allocation block
            _p_graph->pfor_nodes([&](const NodeID u) {
                _next_partition[u] = 0;
                _gains[u]          = 0;
            });
        }

        if (_block_weight_deltas.size() < _p_ctx->k) {
            _block_weight_deltas.resize(_p_ctx->k);
            _p_graph->pfor_blocks([&](const BlockID b) { _block_weight_deltas[b] = 0; });
        }
    };

    TIMED_SCOPE("Initialization") {
        _p_graph->pfor_nodes([&](const NodeID u) {
            _next_partition[u] = _p_graph->block(_color_sorted_nodes[u]);
            _gains[u]          = 0;
        });
    };

    for (std::size_t iter = 0; iter < _input_ctx.refinement.lp.num_iterations; ++iter) {
        NodeID num_moves = 0;
        for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
            num_moves += find_moves(c);
            perform_moves(c);
        }

        // Abort early if there were no moves during a full pass
        MPI_Allreduce(MPI_IN_PLACE, &num_moves, 1, mpi::type::get<NodeID>(), MPI_SUM, _p_graph->communicator());

        const EdgeWeight current_cut       = IFSTATS(metrics::edge_cut(*_p_graph));
        const double     current_imbalance = IFSTATS(metrics::imbalance(*_p_graph));
        STATS << "Iteration " << iter << ": moved " << num_moves << " nodes, changed edge cut to " << current_cut
              << ", changed imbalance to " << current_imbalance;

        if (num_moves == 0) {
            break;
        }
    }
}

void ColoredLPRefiner::perform_moves(const ColorID c) {
    SCOPED_TIMER("Perform moves");

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    const auto block_gains = TIMED_SCOPE("Gather block gain and block weight gain values") {
        if (_input_ctx.refinement.lp.ignore_probabilities) {
            return BlockGainsContainer{};
        }

        parallel::vector_ets<EdgeWeight>  block_gains_ets(_p_ctx->k);
        parallel::vector_ets<BlockWeight> block_weight_gains_ets(_p_ctx->k);

        _p_graph->pfor_nodes_range(seq_from, seq_to, [&](const auto& r) {
            auto& block_gains = block_gains_ets.local();

            for (NodeID seq_u = r.begin(); seq_u != r.end(); ++seq_u) {
                const NodeID  u    = _color_sorted_nodes[seq_u];
                const BlockID from = _p_graph->block(u);
                const BlockID to   = _next_partition[seq_u];
                if (from != to) {
                    block_gains[to] += _gains[seq_u];
                }
            }
        });

        auto block_gains = block_gains_ets.combine(std::plus{});

        MPI_Allreduce(
            MPI_IN_PLACE, block_gains.data(), asserting_cast<int>(_p_ctx->k), mpi::type::get<EdgeWeight>(), MPI_SUM,
            _p_graph->communicator()
        );

        return block_gains;
    };

    TIMED_SCOPE("Perform moves") {
        // Get global block weight deltas to decide if we can execute some moves right away
        MPI_Allreduce(
            MPI_IN_PLACE, _block_weight_deltas.data(), asserting_cast<int>(_p_ctx->k), mpi::type::get<BlockWeight>(),
            MPI_SUM, _p_graph->communicator()
        );

        for (std::size_t i = 0; i < _input_ctx.refinement.lp.num_move_attempts; ++i) {
            if (attempt_moves(c, block_gains)) {
                break;
            }
        }

        synchronize_state(c);
    };

    // Reset _next_partition for next round
    TIMED_SCOPE("Reset arrays") {
        _p_graph->pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
            const NodeID u         = _color_sorted_nodes[seq_u];
            _next_partition[seq_u] = _p_graph->block(u);
        });
        _p_graph->pfor_blocks([&](const BlockID b) { _block_weight_deltas[b] = 0; });
    };
}

bool ColoredLPRefiner::attempt_moves(const ColorID c, const BlockGainsContainer& block_gains) {
    struct Move {
        Move(const NodeID seq_u, const NodeID u, const BlockID from) : seq_u(seq_u), u(u), from(from) {}
        NodeID  seq_u;
        NodeID  u;
        BlockID from;
    };

    // Keep track of the moves that we perform so that we can roll back in case the probabilistic moves made the
    // partition imbalanced
    tbb::concurrent_vector<Move> moves;

    // Track change in block weights to determine whether the partition became imbalanced
    NoinitVector<BlockWeight> block_weight_deltas(_p_ctx->k);
    tbb::parallel_for<BlockID>(0, _p_ctx->k, [&](const BlockID b) { block_weight_deltas[b] = 0; });

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    _p_graph->pfor_nodes_range(seq_from, seq_to, [&](const auto& r) {
        auto& rand = Random::instance();

        for (NodeID seq_u = r.begin(); seq_u != r.end(); ++seq_u) {
            const NodeID u = _color_sorted_nodes[seq_u];

            // Only iterate over nodes that changed block
            if (_next_partition[seq_u] == _p_graph->block(u) || _next_partition[seq_u] == kInvalidBlockID) {
                continue;
            }

            // Compute move probability and perform it
            // Or always perform the move if move probabilities are disabled
            const BlockID to          = _next_partition[seq_u];
            const double  probability = [&] {
                if (_input_ctx.refinement.lp.ignore_probabilities
                    || _p_graph->block_weight(to) + _block_weight_deltas[to] <= _p_ctx->graph.max_block_weight(to)) {
                    return 1.0;
                }

                const double      gain_prob = (block_gains[to] == 0) ? 1.0 : 1.0 * _gains[seq_u] / block_gains[to];
                const BlockWeight residual_block_weight =
                    _p_ctx->graph.max_block_weight(to) - _p_graph->block_weight(to);
                return gain_prob * residual_block_weight / _p_graph->node_weight(u);
            }();

            if (_input_ctx.refinement.lp.ignore_probabilities || rand.random_bool(probability)) {
                const BlockID    from     = _p_graph->block(u);
                const NodeWeight u_weight = _p_graph->node_weight(u);

                moves.emplace_back(seq_u, u, from);
                __atomic_fetch_sub(&block_weight_deltas[from], u_weight, __ATOMIC_RELAXED);
                __atomic_fetch_add(&block_weight_deltas[to], u_weight, __ATOMIC_RELAXED);
                _p_graph->set_block<false>(u, to);

                // Temporary mark that this node was actually moved
                // We will revert this during synchronization or on rollback
                _next_partition[seq_u] = kInvalidBlockID;
            }
        }
    });

    // Compute global block weights after moves
    MPI_Allreduce(
        MPI_IN_PLACE, block_weight_deltas.data(), asserting_cast<int>(_p_ctx->k), mpi::type::get<BlockWeight>(),
        MPI_SUM, _p_graph->communicator()
    );

    // Check for balance violations
    parallel::Atomic<std::uint8_t> feasible = 1;
    if (!_input_ctx.refinement.lp.ignore_probabilities) {
        _p_graph->pfor_blocks([&](const BlockID b) {
            // If blocks were already overloaded before refinement, accept it as feasible if their weight did not
            // increase (i.e., delta is <= 0) == first part of this if condition
            if (block_weight_deltas[b] > 0
                && _p_graph->block_weight(b) + block_weight_deltas[b] > _p_ctx->graph.max_block_weight(b)) {
                feasible = 0;
            }
        });
    }

    // Revert moves if resulting partition is infeasible
    // Otherwise, update block weights cached in the graph data structure
    if (!feasible) {
        tbb::parallel_for(moves.range(), [&](const auto r) {
            for (const auto& [seq_u, u, from]: r) {
                _next_partition[seq_u] = _p_graph->block(u);
                _p_graph->set_block<false>(u, from);
            }
        });
    } else {
        _p_graph->pfor_blocks([&](const BlockID b) {
            _p_graph->set_block_weight(b, _p_graph->block_weight(b) + block_weight_deltas[b]);
        });
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
        _p_graph->graph(), seq_from, seq_to,

        // Map sequence index to node
        [&](const NodeID seq_u) { return _color_sorted_nodes[seq_u]; },

        // We set _next_partition[] to kInvalidBlockID for nodes that were moved during perform_moves()
        [&](const NodeID seq_u, NodeID) -> bool { return _next_partition[seq_u] == kInvalidBlockID; },

        // Send move to each ghost node adjacent to u
        [&](const NodeID seq_u, const NodeID u, PEID) -> MoveMessage {
            // perform_moves() marks nodes that were moved locally by setting _next_partition[] to kInvalidBlockID
            // here, we revert this mark
            const BlockID block    = _p_graph->block(u);
            _next_partition[seq_u] = block;
            return {.local_node = u, .new_block = block};
        },

        // Move ghost nodes
        [&](const auto recv_buffer, const PEID pe) {
            tbb::parallel_for(static_cast<std::size_t>(0), recv_buffer.size(), [&](const std::size_t i) {
                const auto [local_node_on_pe, new_block] = recv_buffer[i];
                const GlobalNodeID global_node           = _p_graph->offset_n(pe) + local_node_on_pe;
                const NodeID       local_node            = _p_graph->global_to_local_node(global_node);
                KASSERT(new_block != _p_graph->block(local_node)); // Otherwise, we should not have gotten this message

                _p_graph->set_block<false>(local_node, new_block);
            });
        }
    );
}

NodeID ColoredLPRefiner::find_moves(const ColorID c) {
    SCOPED_TIMER("Find moves");

    const NodeID seq_from = _color_sizes[c];
    const NodeID seq_to   = _color_sizes[c + 1];

    tbb::enumerable_thread_specific<NodeID>                         num_moved_nodes_ets;
    tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> rating_maps_ets([&] {
        return RatingMap<EdgeWeight, BlockID>(_p_ctx->k);
    });

    _p_graph->pfor_nodes_range(seq_from, seq_to, [&](const auto& r) {
        auto& num_moved_nodes = num_moved_nodes_ets.local();
        auto& rating_map      = rating_maps_ets.local();
        auto& random          = Random::instance();

        for (NodeID seq_u = r.begin(); seq_u != r.end(); ++seq_u) {
            const NodeID u = _color_sorted_nodes[seq_u];

            auto action = [&](auto& map) {
                for (const auto [e, v]: _p_graph->neighbors(u)) {
                    const BlockID    b      = _p_graph->block(v);
                    const EdgeWeight weight = _p_graph->edge_weight(e);
                    map[b] += weight;
                }

                const BlockID    u_block     = _p_graph->block(u);
                const NodeWeight u_weight    = _p_graph->node_weight(u);
                EdgeWeight       best_weight = std::numeric_limits<EdgeWeight>::min();
                BlockID          best_block  = u_block;
                for (const auto [block, weight]: map.entries()) {
                    if (_p_graph->block_weight(block) + u_weight > _p_ctx->graph.max_block_weight(block)) {
                        continue;
                    }

                    if (weight > best_weight || (weight == best_weight && random.random_bool())) {
                        best_weight = weight;
                        best_block  = block;
                    }
                }

                if (best_block != u_block) {
                    _gains[seq_u]          = best_weight - map[u_block];
                    _next_partition[seq_u] = best_block;
                    _block_weight_deltas[u_block] -= u_weight;
                    _block_weight_deltas[best_block] += u_weight;
                    ++num_moved_nodes;
                }

                map.clear();
            };

            rating_map.update_upper_bound_size(std::min<BlockID>(_p_ctx->k, _p_graph->degree(u)));
            rating_map.run_with_map(action, action);
        }
    });

    return num_moved_nodes_ets.combine(std::plus{});
}
} // namespace kaminpar::dist
