/*******************************************************************************
 * Distributed label propagation refiner that uses a graph coloring to avoid
 * move conflicts.
 *
 * @file:   clp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   09.11.2022
 ******************************************************************************/
#include "kaminpar-dist/refinement/lp/clp_refiner.h"

#include <algorithm>
#include <unordered_map>

#include <mpi.h>
#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/metrics.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/vector_ets.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
ColoredLPRefinerFactory::ColoredLPRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner> ColoredLPRefinerFactory::create(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  return std::make_unique<ColoredLPRefiner>(_ctx, p_graph, p_ctx);
}

ColoredLPRefiner::ColoredLPRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _input_ctx(ctx),
      _ctx(ctx.refinement.colored_lp),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _gain_statistics() {}

void ColoredLPRefiner::initialize() {
  const DistributedGraph &graph = _p_graph.graph();

  mpi::barrier(graph.communicator());

  SCOPED_TIMER("Colored LP refinement");
  SCOPED_TIMER("Initialization");

  const auto coloring = [&] {
    // Graph is already sorted by a coloring -> reconstruct this coloring
    // @todo if we always want to do this, optimize this refiner
    if (graph.color_sorted()) {
      LOG << "Graph sorted by colors: using precomputed coloring";

      NoinitVector<ColorID> coloring(graph.n()
      ); // We do not actually need the colors for ghost nodes

      // @todo parallelize
      NodeID pos = 0;
      for (ColorID c = 0; c < graph.number_of_colors(); ++c) {
        const std::size_t size = graph.color_size(c);
        std::fill(coloring.begin() + pos, coloring.begin() + pos + size, c);
        pos += size;
      }

      return coloring;
    }

    // Otherwise, compute a coloring now
    LOG << "Computing new coloring";
    return compute_node_coloring_sequentially(
        graph, _ctx.coloring_chunks.compute(_input_ctx.parallel)
    );
  }();

  const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
  const ColorID num_colors = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());
  STATS << "Number of colors: " << num_colors;

  TIMED_SCOPE("Allocation") {
    _color_sorted_nodes.resize(graph.n());
    _color_sizes.resize(num_colors + 1);
    _color_blacklist.resize(num_colors);
    tbb::parallel_for<std::size_t>(0, _color_sorted_nodes.size(), [&](const std::size_t i) {
      _color_sorted_nodes[i] = 0;
    });
    tbb::parallel_for<std::size_t>(0, _color_sizes.size(), [&](const std::size_t i) {
      _color_sizes[i] = 0;
    });
    tbb::parallel_for<std::size_t>(0, _color_blacklist.size(), [&](const std::size_t i) {
      _color_blacklist[i] = 0;
    });

    if (_ctx.use_active_set) {
      _is_active.resize(graph.total_n());
      graph.pfor_all_nodes([&](const NodeID u) { _is_active[u] = 1; });
    }
  };

  TIMED_SCOPE("Count color sizes") {
    if (graph.color_sorted()) {
      const auto &color_sizes = graph.get_color_sizes();
      _color_sizes.assign(color_sizes.begin(), color_sizes.end());
    } else {
      graph.pfor_nodes([&](const NodeID u) {
        const ColorID c = coloring[u];
        KASSERT(c < num_colors);
        __atomic_fetch_add(&_color_sizes[c], 1, __ATOMIC_RELAXED);
      });
      parallel::prefix_sum(_color_sizes.begin(), _color_sizes.end(), _color_sizes.begin());
    }
  };

  TIMED_SCOPE("Sort nodes") {
    if (graph.color_sorted()) {
      // @todo parallelize
      std::iota(_color_sorted_nodes.begin(), _color_sorted_nodes.end(), 0);
    } else {
      graph.pfor_nodes([&](const NodeID u) {
        const ColorID c = coloring[u];
        const std::size_t i = __atomic_sub_fetch(&_color_sizes[c], 1, __ATOMIC_SEQ_CST);
        KASSERT(i < _color_sorted_nodes.size());
        _color_sorted_nodes[i] = u;
      });
    }
  };

  TIMED_SCOPE("Compute color blacklist") {
    if (_ctx.small_color_blacklist == 0 ||
        (_ctx.only_blacklist_input_level && graph.global_n() != _input_ctx.partition.graph->global_n
        )) {
      STATS << "Do not blacklist any colors";
      return;
    }

    NoinitVector<GlobalNodeID> global_color_sizes(num_colors);
    tbb::parallel_for<ColorID>(0, num_colors, [&](const ColorID c) {
      global_color_sizes[c] = _color_sizes[c + 1] - _color_sizes[c];
    });
    MPI_Allreduce(
        MPI_IN_PLACE,
        global_color_sizes.data(),
        asserting_cast<int>(num_colors),
        mpi::type::get<GlobalNodeID>(),
        MPI_SUM,
        graph.communicator()
    );

    // @todo parallelize the rest of this section
    std::vector<ColorID> sorted_by_size(num_colors);
    std::iota(sorted_by_size.begin(), sorted_by_size.end(), 0);
    std::sort(
        sorted_by_size.begin(),
        sorted_by_size.end(),
        [&](const ColorID lhs, const ColorID rhs) {
          return global_color_sizes[lhs] < global_color_sizes[rhs];
        }
    );

    GlobalNodeID excluded_so_far = 0;
    for (const ColorID c : sorted_by_size) {
      excluded_so_far += global_color_sizes[c];
      const double percentage = 1.0 * excluded_so_far / graph.global_n();
      if (percentage <= _ctx.small_color_blacklist) {
        _color_blacklist[c] = 1;
      } else {
        break;
      }
    }
  };

  KASSERT(_color_sizes.front() == 0u);
  KASSERT(_color_sizes.back() == graph.n());

  IFSTATS(_gain_statistics.initialize(num_colors));
}

bool ColoredLPRefiner::refine() {
  mpi::barrier(_p_graph.communicator());

  SCOPED_TIMER("Colored LP refinement");
  SCOPED_TIMER("Refinement");

  TIMED_SCOPE("Allocation") {
    KASSERT(_next_partition.size() == _gains.size());
    if (_next_partition.size() < _p_graph.n()) {
      _next_partition.resize(_p_graph.n());
      _gains.resize(_p_graph.n());

      // Attribute first touch running time to allocation block
      _p_graph.pfor_nodes([&](const NodeID u) {
        _next_partition[u] = 0;
        _gains[u] = 0;
      });
    }

    if (_ctx.track_local_block_weights && _block_weight_deltas.size() < _p_ctx.k) {
      _block_weight_deltas.resize(_p_ctx.k);
      _p_graph.pfor_blocks([&](const BlockID b) { _block_weight_deltas[b] = 0; });
    }
  };

  TIMED_SCOPE("Initialization") {
    _p_graph.pfor_nodes([&](const NodeID u) {
      _next_partition[u] = _p_graph.block(_color_sorted_nodes[u]);
      _gains[u] = 0;
    });
  };

  for (int iter = 0; iter < _ctx.num_iterations; ++iter) {
    NodeID num_found_moves = 0;
    NodeID num_performed_moves = 0;
    for (ColorID c = 0; c + 1 < _color_sizes.size(); ++c) {
      if (_color_blacklist[c]) {
        continue;
      }

      num_found_moves += find_moves(c);
      for (int round = 0; round < _ctx.num_move_execution_iterations; ++round) {
        num_performed_moves += perform_moves(c);
      }

      // Reset arrays for next round
      const NodeID seq_from = _color_sizes[c];
      const NodeID seq_to = _color_sizes[c + 1];
      _p_graph.pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
        const NodeID u = _color_sorted_nodes[seq_u];
        _next_partition[seq_u] = _p_graph.block(u);
      });
      if (_ctx.track_local_block_weights) {
        _p_graph.pfor_blocks([&](const BlockID b) { _block_weight_deltas[b] = 0; });
      }
    }

    // Abort early if there were no moves during a full pass
    MPI_Allreduce(
        MPI_IN_PLACE,
        &num_found_moves,
        1,
        mpi::type::get<NodeID>(),
        MPI_SUM,
        _p_graph.communicator()
    );
    IFSTATS(MPI_Allreduce(
        MPI_IN_PLACE,
        &num_performed_moves,
        1,
        mpi::type::get<NodeID>(),
        MPI_SUM,
        _p_graph.communicator()
    ));

    STATS << "Iteration " << iter << ": found " << num_found_moves << " moves, performed "
          << num_performed_moves << " moves";

    /*
    const EdgeWeight current_cut       = IFSTATS(metrics::edge_cut(*_p_graph));
    const double     current_imbalance = IFSTATS(metrics::imbalance(*_p_graph));
    STATS << "Iteration " << iter << ": found " << num_found_moves << " moves,
    performed " << num_performed_moves
          << " moves, changed edge cut to " << current_cut << ", changed
    imbalance to " << current_imbalance;
    */

    if (num_found_moves == 0) {
      break;
    }
  }

  IFSTATS(_gain_statistics.summarize_by_size(_color_sizes, _p_graph.communicator()));
  mpi::barrier(_p_graph.communicator());

  return false;
}

NodeID ColoredLPRefiner::perform_moves(const ColorID c) {
  switch (_ctx.move_execution_strategy) {
  case LabelPropagationMoveExecutionStrategy::PROBABILISTIC:
    return perform_probabilistic_moves(c);

  case LabelPropagationMoveExecutionStrategy::BEST_MOVES:
    return perform_best_moves(c);

  case LabelPropagationMoveExecutionStrategy::LOCAL_MOVES:
    return perform_local_moves(c);

  default:
    KASSERT(false, "", assert::always);
    return 0;
  }
}

NodeID ColoredLPRefiner::perform_best_moves(const ColorID c) {
  SCOPED_TIMER("Perform moves");

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  // Collect nodes that changed block
  // @todo could be collected right away
  // @todo parallelize
  std::vector<MoveCandidate> move_candidates;
  for (const NodeID seq_u : _p_graph.nodes(seq_from, seq_to)) {
    const NodeID u = _color_sorted_nodes[seq_u];
    const BlockID from = _p_graph.block(u);
    const BlockID to = _next_partition[seq_u];
    if (to != kInvalidBlockID && to != from) {
      const GlobalNodeID global_u = _p_graph.local_to_global_node(u);
      const EdgeWeight gain = _gains[seq_u];
      const NodeWeight weight = _p_graph.node_weight(u);

      move_candidates.push_back({seq_u, global_u, from, to, gain, weight});
    }
  }

  // reduce_move_candidates requires candidates to be sorted by their `from`
  // block
  std::sort(move_candidates.begin(), move_candidates.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.from < rhs.from;
  });

  // Binary reduction tree to find the best candidates for each block, globally
  move_candidates = reduce_move_candidates(std::move(move_candidates));

  const std::size_t num_candidates = mpi::bcast(move_candidates.size(), 0, _p_graph.communicator());
  move_candidates.resize(num_candidates); // No effect on PE 0 == root
  mpi::bcast(
      move_candidates.data(), asserting_cast<int>(num_candidates), 0, _p_graph.communicator()
  );

  // Move nodes
  NodeID num_local_moved_nodes = 0;
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  EdgeWeight expected_gain_improvement = 0;
  const EdgeWeight edge_cut_before = metrics::edge_cut(_p_graph);
#endif

  for (const MoveCandidate &candidate : move_candidates) {
    if (_p_graph.contains_global_node(candidate.node)) {
      const NodeID local_node = _p_graph.global_to_local_node(candidate.node);
      _p_graph.set_block<false>(local_node, candidate.to);
      if (local_node < _p_graph.n()) {
        activate_neighbors(local_node);
        ++num_local_moved_nodes;
        _next_partition[candidate.local_seq] = kInvalidBlockID;
        IFSTATS(_gain_statistics.record_gain(candidate.gain, c));
      }
    }
    _p_graph.set_block_weight(candidate.to, _p_graph.block_weight(candidate.to) + candidate.weight);
    _p_graph.set_block_weight(
        candidate.from, _p_graph.block_weight(candidate.from) - candidate.weight
    );

#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    expected_gain_improvement += candidate.gain;
#endif
  }

  KASSERT(
      debug::validate_partition(_p_graph),
      "invalid partition state after executing node moves",
      assert::heavy
  );
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  const EdgeWeight edge_cut_after = metrics::edge_cut(_p_graph);
  KASSERT(
      edge_cut_before - edge_cut_after == expected_gain_improvement,
      "edge cut change inconsistent with move gains"
  );
#endif

  return num_local_moved_nodes;
}

auto ColoredLPRefiner::reduce_move_candidates(std::vector<MoveCandidate> &&candidates)
    -> std::vector<MoveCandidate> {
  const int size = mpi::get_comm_size(_p_graph.communicator());
  const int rank = mpi::get_comm_rank(_p_graph.communicator());
  KASSERT(math::is_power_of_2(size), "#PE must be a power of two", assert::always);

  int active_size = size;
  while (active_size > 1) {
    // false = receiver
    // true = sender
    const bool role = (rank >= active_size / 2);

    if (role) {
      const PEID dest = rank - active_size / 2;
      mpi::send(candidates.data(), candidates.size(), dest, 0, _p_graph.communicator());
      break;
    } else {
      const PEID src = rank + active_size / 2;
      auto tmp_buffer = mpi::probe_recv<MoveCandidate, std::vector<MoveCandidate>>(
          src, 0, _p_graph.communicator()
      );
      candidates = reduce_move_candidates(std::move(candidates), std::move(tmp_buffer));
    }

    active_size /= 2;
  }

  return std::move(candidates);
}

auto ColoredLPRefiner::reduce_move_candidates(
    std::vector<MoveCandidate> &&a, std::vector<MoveCandidate> &&b
) -> std::vector<MoveCandidate> {
  std::vector<MoveCandidate> ans;

  // precondition: candidates are sorted by from block
  KASSERT([&] {
    for (std::size_t i = 1; i < a.size(); ++i) {
      KASSERT(a[i].from >= a[i - 1].from);
    }
    for (std::size_t i = 1; i < b.size(); ++i) {
      KASSERT(b[i].from >= b[i - 1].from);
    }
    return true;
  }());

  std::size_t i = 0; // index in a
  std::size_t j = 0; // index in b

  // Reset _block_weight_deltas
  tbb::parallel_for<BlockID>(0, _p_ctx.k, [&](const BlockID b) { _block_weight_deltas[b] = 0; });

  auto try_add_candidate = [&](std::vector<MoveCandidate> &ans, const MoveCandidate &candidate) {
    if (_p_graph.block_weight(candidate.to) + _block_weight_deltas[candidate.to] +
            candidate.weight <=
        _p_ctx.graph->max_block_weight(candidate.to)) {
      ans.push_back(candidate);
      _block_weight_deltas[candidate.to] += candidate.weight;
    }
  };

  // For each target block, find the highest gain prefix
  for (i = 0, j = 0; i < a.size() && j < b.size();) {
    const BlockID from = std::min<BlockID>(a[i].from, b[j].from);

    // Find region in `a` and `b` with nodes from `from`
    std::size_t i_end = i;
    std::size_t j_end = j;
    while (i_end < a.size() && a[i_end].from == from) {
      ++i_end;
    }
    while (j_end < b.size() && b[j_end].from == from) {
      ++j_end;
    }

    // Merge lists and sort by relative gain
    const std::size_t num_in_a = i_end - i;
    const std::size_t num_in_b = j_end - j;
    const std::size_t num = num_in_a + num_in_b;

    std::vector<MoveCandidate> candidates(num);
    std::copy(a.begin() + i, a.begin() + i_end, candidates.begin());
    std::copy(b.begin() + j, b.begin() + j_end, candidates.begin() + num_in_a);

    if (_ctx.sort_by_rel_gain) {
      std::sort(candidates.begin(), candidates.end(), [&](const auto &lhs, const auto &rhs) {
        const double lhs_rel_gain = 1.0 * lhs.gain / lhs.weight;
        const double rhs_rel_gain = 1.0 * rhs.gain / rhs.weight;
        return lhs_rel_gain > rhs_rel_gain || (lhs_rel_gain == rhs_rel_gain && lhs.node > rhs.node);
      });
    } else {
      std::sort(candidates.begin(), candidates.end(), [&](const auto &lhs, const auto &rhs) {
        return lhs.gain > rhs.gain || (lhs.gain == rhs.gain && lhs.node > rhs.node);
      });
    }

    for (NodeID candidate = 0; candidate < candidates.size(); ++candidate) {
      try_add_candidate(ans, candidates[candidate]);
    }

    // Move forward
    i = i_end;
    j = j_end;
  }

  // Keep remaining moves
  for (; i < a.size(); ++i) {
    try_add_candidate(ans, a[i]);
  }
  for (; j < b.size(); ++j) {
    try_add_candidate(ans, b[j]);
  }

  return ans;
}

NodeID ColoredLPRefiner::perform_local_moves(const ColorID c) {
  SCOPED_TIMER("Perform moves");

  KASSERT(
      _ctx.track_local_block_weights,
      "enable block weight tracking to use this move execution strategy",
      assert::always
  );

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;
  _p_graph.pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
    const NodeID u = _color_sorted_nodes[seq_u];
    const BlockID to = _next_partition[seq_u];
    KASSERT(to != kInvalidBlockID);

    if (to != _p_graph.block(u)) {
      activate_neighbors(u);
      _next_partition[seq_u] = kInvalidNodeID; // Mark as moved
      _p_graph.set_block<false>(u, to);
      ++num_moved_nodes_ets.local();
      IFSTATS(_gain_statistics.record_gain(_gains[seq_u], c));
    }
  });

  synchronize_state(c);

  // Reset _next_partition[.] state such that a second call to this function has
  // no effect I.e., multiple commit rounds have no effects with this strategy
  _p_graph.pfor_nodes(seq_from, seq_to, [&](const NodeID seq_u) {
    const NodeID u = _color_sorted_nodes[seq_u];
    _next_partition[seq_u] = _p_graph.block(u);
  });

  TIMED_SCOPE("Update block weights") {
    MPI_Allreduce(
        MPI_IN_PLACE,
        _block_weight_deltas.data(),
        asserting_cast<int>(_p_ctx.k),
        mpi::type::get<BlockWeight>(),
        MPI_SUM,
        _p_graph.communicator()
    );
    _p_graph.pfor_blocks([&](const BlockID b) {
      _p_graph.set_block_weight(b, _p_graph.block_weight(b) + _block_weight_deltas[b]);
    });
  };

  return num_moved_nodes_ets.combine(std::plus{});
}

NodeID ColoredLPRefiner::perform_probabilistic_moves(const ColorID c) {
  SCOPED_TIMER("Perform moves");

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  const auto block_gains = TIMED_SCOPE("Gather block gain and block weight gain values") {
    parallel::vector_ets<EdgeWeight> block_gains_ets(_p_ctx.k);
    parallel::vector_ets<BlockWeight> block_weight_gains_ets(_p_ctx.k);

    _p_graph.pfor_nodes_range(seq_from, seq_to, [&](const auto &r) {
      auto &block_gains = block_gains_ets.local();

      for (NodeID seq_u = r.begin(); seq_u != r.end(); ++seq_u) {
        const NodeID u = _color_sorted_nodes[seq_u];
        const BlockID from = _p_graph.block(u);
        const BlockID to = _next_partition[seq_u];
        if (to != from && to != kInvalidBlockID) {
          block_gains[to] += _gains[seq_u];
        }
      }
    });

    auto block_gains = block_gains_ets.combine(std::plus{});

    MPI_Allreduce(
        MPI_IN_PLACE,
        block_gains.data(),
        asserting_cast<int>(_p_ctx.k),
        mpi::type::get<EdgeWeight>(),
        MPI_SUM,
        _p_graph.communicator()
    );

    return block_gains;
  };

  NodeID num_performed_moves = 0;
  for (int i = 0; i < _ctx.num_probabilistic_move_attempts; ++i) {
    num_performed_moves = try_probabilistic_moves(c, block_gains);
    if (num_performed_moves == kInvalidNodeID) {
      num_performed_moves = 0;
    } else {
      break;
    }
  }

  return num_performed_moves;
}

NodeID
ColoredLPRefiner::try_probabilistic_moves(const ColorID c, const BlockGainsContainer &block_gains) {
  struct Move {
    Move(const NodeID seq_u, const NodeID u, const BlockID from) : seq_u(seq_u), u(u), from(from) {}
    NodeID seq_u;
    NodeID u;
    BlockID from;
  };

  // Keep track of the moves that we perform so that we can roll back in case
  // the probabilistic moves made the partition imbalanced
  tbb::concurrent_vector<Move> moves;

  // Track change in block weights to determine whether the partition became
  // imbalanced
  NoinitVector<BlockWeight> block_weight_deltas(_p_ctx.k);
  tbb::parallel_for<BlockID>(0, _p_ctx.k, [&](const BlockID b) { block_weight_deltas[b] = 0; });

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  EdgeWeight total_gain = 0;
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
  const GlobalEdgeWeight edge_cut_before = metrics::edge_cut(_p_graph);
#endif

  tbb::enumerable_thread_specific<NodeID> num_performed_moves_ets;
  _p_graph.pfor_nodes_range(seq_from, seq_to, [&](const auto &r) {
    auto &rand = Random::instance();
    auto &num_performed_moves = num_performed_moves_ets.local();

    for (NodeID seq_u = r.begin(); seq_u != r.end(); ++seq_u) {
      const NodeID u = _color_sorted_nodes[seq_u];

      // Only iterate over nodes that changed block
      if (_next_partition[seq_u] == _p_graph.block(u) ||
          _next_partition[seq_u] == kInvalidBlockID) {
        continue;
      }

      // Compute move probability and perform it
      // Or always perform the move if move probabilities are disabled
      const BlockID to = _next_partition[seq_u];
      const double probability = [&] {
        const double gain_prob =
            (block_gains[to] == 0) ? 1.0 : 1.0 * _gains[seq_u] / block_gains[to];
        const BlockWeight residual_block_weight =
            _p_ctx.graph->max_block_weight(to) - _p_graph.block_weight(to);
        return gain_prob * residual_block_weight / _p_graph.node_weight(u);
      }();

      if (rand.random_bool(probability)) {
        const BlockID from = _p_graph.block(u);
        const NodeWeight u_weight = _p_graph.node_weight(u);

        moves.emplace_back(seq_u, u, from);
        __atomic_fetch_sub(&block_weight_deltas[from], u_weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&block_weight_deltas[to], u_weight, __ATOMIC_RELAXED);
        _p_graph.set_block<false>(u, to);
        ++num_performed_moves;

        // Temporary mark that this node was actually moved
        // We will revert this during synchronization or on rollback
        _next_partition[seq_u] = kInvalidBlockID;

        // Record total gain for assertions and statistics
        total_gain += _gains[seq_u];
      }
    }
  });

  const NodeID num_performed_moves = num_performed_moves_ets.combine(std::plus{});

  // Compute global block weights after moves
  MPI_Allreduce(
      MPI_IN_PLACE,
      block_weight_deltas.data(),
      asserting_cast<int>(_p_ctx.k),
      mpi::type::get<BlockWeight>(),
      MPI_SUM,
      _p_graph.communicator()
  );

  // Check for balance violations
  parallel::Atomic<std::uint8_t> feasible = 1;
  _p_graph.pfor_blocks([&](const BlockID b) {
    // If blocks were already overloaded before refinement, accept it as
    // feasible if their weight did not increase (i.e., delta is <= 0) == first
    // part of this if condition
    if (block_weight_deltas[b] > 0 &&
        _p_graph.block_weight(b) + block_weight_deltas[b] > _p_ctx.graph->max_block_weight(b)) {
      feasible = 0;
    }
  });

  // Revert moves if resulting partition is infeasible
  // Otherwise, update block weights cached in the graph data structure
  if (!feasible) {
    tbb::parallel_for(moves.range(), [&](const auto r) {
      for (const auto &[seq_u, u, from] : r) {
        _next_partition[seq_u] = _p_graph.block(u);
        _p_graph.set_block<false>(u, from);
      }
    });
  } else {
    synchronize_state(c);
    _p_graph.pfor_blocks([&](const BlockID b) {
      _p_graph.set_block_weight(b, _p_graph.block_weight(b) + block_weight_deltas[b]);
    });

    // Revert mark in _next_partition[.] for next commit round
    tbb::parallel_for(moves.range(), [&](const auto r) {
      for (const auto &[seq_u, u, from] : r) {
        _next_partition[seq_u] = _p_graph.block(u);
        activate_neighbors(u);
      }
    });

    // Check that the partition improved as expected
#if KASSERT_ENABLED(ASSERTION_LEVEL_NORMAL)
    const GlobalEdgeWeight global_expected_total_gain =
        mpi::allreduce<GlobalEdgeWeight>(total_gain, MPI_SUM, _p_graph.communicator());
    const GlobalEdgeWeight edge_cut_after = metrics::edge_cut(_p_graph);
    KASSERT(
        edge_cut_before - edge_cut_after == global_expected_total_gain,
        "sum of individual move gains does not equal the change in edge cut"
    );
#endif

    IFSTATS(_gain_statistics.record_gain(total_gain, c));
  }

  mpi::barrier(_p_graph.communicator());
  return feasible ? num_performed_moves : kInvalidNodeID;
}

void ColoredLPRefiner::synchronize_state(const ColorID c) {
  struct MoveMessage {
    NodeID local_node;
    BlockID new_block;
  };

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  mpi::graph::sparse_alltoall_interface_to_pe_custom_range<MoveMessage>(
      _p_graph.graph(),
      seq_from,
      seq_to,

      // Map sequence index to node
      [&](const NodeID seq_u) { return _color_sorted_nodes[seq_u]; },

      // We set _next_partition[] to kInvalidBlockID for nodes that were moved
      // during perform_moves()
      [&](const NodeID seq_u, NodeID) -> bool { return _next_partition[seq_u] == kInvalidBlockID; },

      // Send move to each ghost node adjacent to u
      [&](const NodeID seq_u, const NodeID u, PEID) -> MoveMessage {
        // perform_moves() marks nodes that were moved locally by setting
        // _next_partition[] to kInvalidBlockID here, we revert this mark
        const BlockID block = _p_graph.block(u);
        _next_partition[seq_u] = block;
        return {.local_node = u, .new_block = block};
      },

      // Move ghost nodes
      [&](const auto recv_buffer, const PEID pe) {
        tbb::parallel_for(
            static_cast<std::size_t>(0),
            recv_buffer.size(),
            [&](const std::size_t i) {
              const auto [local_node_on_pe, new_block] = recv_buffer[i];
              const GlobalNodeID global_node = _p_graph.offset_n(pe) + local_node_on_pe;
              const NodeID local_node = _p_graph.global_to_local_node(global_node);
              KASSERT(new_block != _p_graph.block(local_node)); // Otherwise, we should not
                                                                // have gotten this message

              _p_graph.set_block<false>(local_node, new_block);
            }
        );
      }
  );
}

NodeID ColoredLPRefiner::find_moves(const ColorID c) {
  SCOPED_TIMER("Find moves");

  const NodeID seq_from = _color_sizes[c];
  const NodeID seq_to = _color_sizes[c + 1];

  tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;
  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> rating_maps_ets([&] {
    return RatingMap<EdgeWeight, BlockID>(_p_ctx.k);
  });

  auto &graph = _p_graph.graph();
  graph.pfor_nodes_range(seq_from, seq_to, [&](const auto &r) {
    auto &num_moved_nodes = num_moved_nodes_ets.local();
    auto &rating_map = rating_maps_ets.local();
    auto &random = Random::instance();

    for (NodeID seq_u = r.begin(); seq_u != r.end(); ++seq_u) {
      const NodeID u = _color_sorted_nodes[seq_u];

      if (_ctx.use_active_set && !_is_active[u]) {
        continue;
      }

      auto action = [&](auto &map) {
        bool is_interface_node = false;
        for (const auto [e, v] : graph.neighbors(u)) {
          const BlockID b = _p_graph.block(v);
          const EdgeWeight weight = graph.edge_weight(e);
          map[b] += weight;
          is_interface_node |= graph.is_ghost_node(v);
        }

        const BlockID u_block = _p_graph.block(u);
        const NodeWeight u_weight = graph.node_weight(u);
        EdgeWeight best_weight = std::numeric_limits<EdgeWeight>::min();
        BlockID best_block = u_block;
        for (const auto [block, weight] : map.entries()) {
          KASSERT(block < _p_graph.k());

          if (block != u_block) {
            if ((_ctx.track_local_block_weights &&
                 _p_graph.block_weight(block) + _block_weight_deltas[block] + u_weight >
                     _p_ctx.graph->max_block_weight(block)) ||
                (!_ctx.track_local_block_weights &&
                 _p_graph.block_weight(block) + u_weight > _p_ctx.graph->max_block_weight(block))) {
              continue;
            }
          }

          if (weight > best_weight || (weight == best_weight && random.random_bool())) {
            best_weight = weight;
            best_block = block;
          }
        }

        if (best_block != u_block) {
          _gains[seq_u] = best_weight - map[u_block];
          KASSERT(_gains[seq_u] >= 0);

          _next_partition[seq_u] = best_block;
          ++num_moved_nodes;

          if (_ctx.track_local_block_weights) {
            _block_weight_deltas[u_block] -= u_weight;
            _block_weight_deltas[best_block] += u_weight;
          }
        }

        if (_ctx.use_active_set && !is_interface_node) {
          _is_active[u] = 0;
        }

        map.clear();
      };

      rating_map.execute(std::min<BlockID>(_p_ctx.k, _p_graph.degree(u)), action);
    }
  });

  mpi::barrier(_p_graph.communicator());
  return num_moved_nodes_ets.combine(std::plus{});
}

void ColoredLPRefiner::activate_neighbors(const NodeID u) {
  if (!_ctx.use_active_set) {
    return;
  }

  for (const auto &[e, v] : _p_graph.neighbors(u)) {
    _is_active[v] = 1;
  }
}

void ColoredLPRefiner::GainStatistics::initialize(const ColorID c) {
  _gain_per_color.resize(c);
}

void ColoredLPRefiner::GainStatistics::record_gain(const EdgeWeight gain, const ColorID c) {
  KASSERT(!_gain_per_color.empty(), "must call initialize() first");
  KASSERT(c < _gain_per_color.size());
  _gain_per_color[c] += gain;
}

void ColoredLPRefiner::GainStatistics::summarize_by_size(
    const NoinitVector<NodeID> &color_sizes, MPI_Comm comm
) const {
  KASSERT(!_gain_per_color.empty(), "must call initialize() first");
  KASSERT(_gain_per_color.size() <= color_sizes.size());

  std::vector<EdgeWeight> gain_per_color_global(_gain_per_color.begin(), _gain_per_color.end());
  MPI_Allreduce(
      MPI_IN_PLACE,
      gain_per_color_global.data(),
      asserting_cast<int>(gain_per_color_global.size()),
      mpi::type::get<EdgeWeight>(),
      MPI_SUM,
      comm
  );
  const EdgeWeight total_gain =
      parallel::accumulate(gain_per_color_global.begin(), gain_per_color_global.end(), 0);

  std::vector<GlobalNodeID> global_color_sizes(_gain_per_color.size());
  for (ColorID c = 0; c < _gain_per_color.size(); ++c) {
    global_color_sizes[c] = color_sizes[c + 1] - color_sizes[c];
  }
  MPI_Allreduce(
      MPI_IN_PLACE,
      global_color_sizes.data(),
      asserting_cast<int>(global_color_sizes.size()),
      mpi::type::get<GlobalNodeID>(),
      MPI_SUM,
      comm
  );

  // Group by color size
  std::unordered_map<GlobalNodeID, EdgeWeight> gain_by_color_size;
  for (ColorID c = 0; c < gain_per_color_global.size(); ++c) {
    const GlobalNodeID size = global_color_sizes[c];
    const EdgeWeight gain = gain_per_color_global[c];
    gain_by_color_size[size] += gain;
  }

  // Sort by color size
  std::vector<std::pair<GlobalNodeID, EdgeWeight>> gain_by_color_size_sorted;
  for (const auto &[color_size, color_gain] : gain_by_color_size) {
    gain_by_color_size_sorted.emplace_back(color_size, color_gain);
  }
  std::sort(gain_by_color_size_sorted.begin(), gain_by_color_size_sorted.end(), std::greater{});

  EdgeWeight gain_so_far = 0;

  STATS << "Total gain achieved by each color size:";
  for (const auto &[color_size, color_gain] : gain_by_color_size_sorted) {
    gain_so_far += color_gain;
    const double gain_so_far_percentage = 1.0 * gain_so_far / total_gain;
    const double gain_percentage = 1.0 * color_gain / total_gain;

    STATS << "  STATS size=" << color_size << " gain=" << color_gain
          << " percentage=" << gain_percentage << " percentage_so_far=" << gain_so_far_percentage;
  }
}
} // namespace kaminpar::dist
