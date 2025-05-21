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
#include "kaminpar-dist/distributed_label_propagation.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/refinement/lp/lp_stats.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/vector_ets.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {

namespace {

SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(false);

} // namespace

struct LPRefinerConfig : public LabelPropagationConfig {
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, BlockID>;
  using Graph = DistributedGraph;
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;

  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = false;
  static constexpr bool kUseActualGain = true;

  static constexpr bool kUseActiveSetStrategy = false;
  static constexpr bool kUseLocalActiveSetStrategy = true;
};

struct LPRefinerMemoryContext
    : public LabelPropagationMemoryContext<LPRefinerConfig::RatingMap, LPRefinerConfig::ClusterID> {
  StaticArray<BlockID> next_partition;
  StaticArray<EdgeWeight> gains;
  StaticArray<BlockWeight> block_weights;
};

template <typename Graph>
class LPRefinerImpl final
    : public ChunkRandomdLabelPropagation<LPRefinerImpl<Graph>, LPRefinerConfig, Graph> {
  using Base = ChunkRandomdLabelPropagation<LPRefinerImpl<Graph>, LPRefinerConfig, Graph>;
  using Config = LPRefinerConfig;

public:
  explicit LPRefinerImpl(const Context &ctx, const DistributedPartitionedGraph &p_graph)
      : _lp_ctx(ctx.refinement.lp),
        _par_ctx(ctx.parallel) {
    Base::set_max_degree(_lp_ctx.active_high_degree_threshold);
    Base::preinitialize(p_graph.total_n(), p_graph.n(), p_graph.k());
  }

  void setup(LPRefinerMemoryContext &memory_context) {
    Base::setup(memory_context);
    _next_partition = std::move(memory_context.next_partition);
    _gains = std::move(memory_context.gains);
    _block_weights = std::move(memory_context.block_weights);
  }

  LPRefinerMemoryContext release() {
    auto [rating_map_ets, active, favored_clusters] = Base::release();
    return {
        {
            std::move(rating_map_ets),
            std::move(active),
            std::move(favored_clusters),
        },
        std::move(_next_partition),
        std::move(_gains),
        std::move(_block_weights),
    };
  }

  void
  refine(const Graph &graph, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    TIMER_BARRIER(graph.communicator());
    SCOPED_HEAP_PROFILER("Label Propagation");
    SCOPED_TIMER("Label Propagation");

    _p_graph = &p_graph;
    _p_ctx = &p_ctx;

    TIMED_SCOPE("Allocation") {
      if (_next_partition.size() < graph.n()) {
        _next_partition.resize(graph.n());
      }
      if (_gains.size() < graph.n()) {
        _gains.resize(graph.n());
      }
      if (_block_weights.size() < p_graph.k()) {
        _block_weights.resize(p_graph.k());
      }
      Base::allocate();
    };

    Base::initialize(&graph, _p_ctx->k);

    IFSTATS(_stats.reset());
    IFSTATS(_stats.cut_before = metrics::edge_cut(*_p_graph));

    const auto num_chunks = _lp_ctx.chunks.compute(_par_ctx);

    for (int iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      GlobalNodeID num_moved_nodes = 0;

      for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(graph.n(), num_chunks, chunk);
        num_moved_nodes += process_chunk(from, to);
      }

      if (num_moved_nodes == 0) {
        break;
      }
    }

    IFSTATS(_stats.cut_after = metrics::edge_cut(*_p_graph));
    IFSTATS(_stats.print(graph.communicator()));
  }

private:
  GlobalNodeID process_chunk(const NodeID from, const NodeID to) {
    TIMER_BARRIER(_graph->communicator());
    DBG0 << "Running label propagation on chunk [" << from << ".." << to << "]";

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT(ASSERT_NEXT_PARTITION_STATE(), "", assert::heavy);
#endif

    // Run label propagation
    const NodeID num_moved_nodes = TIMED_SCOPE("Local work") {
      return Base::perform_iteration(from, to);
    };

    const auto global_num_moved_nodes =
        mpi::allreduce<GlobalNodeID>(num_moved_nodes, MPI_SUM, _graph->communicator());

    DBG0 << "Moved " << global_num_moved_nodes << " nodes in chunk [" << from << ".." << to << "]";

    if (global_num_moved_nodes == 0) {
      // Nothing to do:
      return 0;
    }

    // Accumulate total weight of nodes moved to each block
    std::vector<BlockWeight> residual_cluster_weights;
    std::vector<EdgeWeight> global_total_gains_to_block;

    TIMED_SCOPE("Gather weights and gains") {
      parallel::vector_ets<BlockWeight> weight_to_block_ets(_p_ctx->k);
      parallel::vector_ets<EdgeWeight> gain_to_block_ets(_p_ctx->k);

      _graph->pfor_nodes_range(from, to, [&](const auto r) {
        auto &weight_to_block = weight_to_block_ets.local();
        auto &gain_to_block = gain_to_block_ets.local();

        for (NodeID u = r.begin(); u < r.end(); ++u) {
          if (_p_graph->block(u) != _next_partition[u]) {
            weight_to_block[_next_partition[u]] += _graph->node_weight(u);
            gain_to_block[_next_partition[u]] += _gains[u];
          }
        }
      });

      const auto weight_to_block = weight_to_block_ets.combine(std::plus{});
      const auto gain_to_block = gain_to_block_ets.combine(std::plus{});
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
    };

    DBG0 << "Performing probabilistic moves ...";

    // Perform probabilistic moves
    for (int i = 0; i < _lp_ctx.num_move_attempts; ++i) {
      if (perform_moves(from, to, residual_cluster_weights, global_total_gains_to_block)) {
        break;
      }
    }

    DBG0 << "Syncing state ...";

    synchronize_state(from, to);
    _graph->pfor_nodes(from, to, [&](const NodeID u) { _next_partition[u] = _p_graph->block(u); });

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
    TIMER_BARRIER(_graph->communicator());
    SCOPED_TIMER("Perform moves");

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    KASSERT(debug::validate_partition(*_p_graph), "", assert::heavy);
#endif

    struct Move {
      Move(const NodeID u, const BlockID from) : u(u), from(from) {}
      NodeID u;
      BlockID from;
    };

    // Perform probabilistic moves, but keep track of moves in case we need to roll back
    StaticArray<BlockWeight> block_weight_deltas(_p_ctx->k);
    tbb::concurrent_vector<Move> moves;

    _graph->pfor_nodes_range(from, to, [&](const auto &r) {
      auto &rand = Random::instance();

      for (NodeID u = r.begin(); u < r.end(); ++u) {
        // Only iterate over nodes that changed block
        if (_next_partition[u] == _p_graph->block(u) || _next_partition[u] == kInvalidBlockID) {
          continue;
        }

        // Compute move probability
        const BlockID b = _next_partition[u];
        const double gain_prob =
            _lp_ctx.ignore_probabilities
                ? 1.0
                : ((total_gains_to_block[b] == 0) ? 1.0
                                                  : 1.0 * _gains[u] / total_gains_to_block[b]);
        const double probability =
            _lp_ctx.ignore_probabilities
                ? 1.0
                : gain_prob *
                      (static_cast<double>(residual_block_weights[b]) / _graph->node_weight(u));
        IFSTATS(_stats.expected_gain += probability * _gains[u]);

        // Perform move with probability
        if (_lp_ctx.ignore_probabilities || rand.random_bool(probability)) {
          IFSTATS(_stats.num_tentatively_moved_nodes++);

          const BlockID from = _p_graph->block(u);
          const BlockID to = _next_partition[u];
          const NodeWeight u_weight = _graph->node_weight(u);

          moves.emplace_back(u, from);
          __atomic_fetch_sub(&block_weight_deltas[from], u_weight, __ATOMIC_RELAXED);
          __atomic_fetch_add(&block_weight_deltas[to], u_weight, __ATOMIC_RELAXED);
          _p_graph->set_block<false>(u, to);

          // Temporary mark that this node was actually move, we will revert this during
          // synchronization or on rollback
          _next_partition[u] = kInvalidBlockID;

          IFSTATS(_stats.realized_gain += _gains[u]);
        } else {
          IFSTATS(_stats.num_tentatively_rejected_nodes++);
          IFSTATS(_stats.rejected_gain += _gains[u]);
        }
      }
    });

    // Compute global block weights after moves
    mpi::inplace_sparse_allreduce(block_weight_deltas, _p_ctx->k, MPI_SUM, _graph->communicator());

    // Check for balance violations
    std::atomic<std::uint8_t> feasible = 1;
    if (!_lp_ctx.ignore_probabilities) {
      _p_graph->pfor_blocks([&](const BlockID b) {
        if (_p_graph->block_weight(b) + block_weight_deltas[b] > max_cluster_weight(b) &&
            block_weight_deltas[b] > 0) {
          feasible = 0;
        }
      });
    }

    // Record statistics
    if constexpr (kStatistics) {
      if (!feasible) {
        _stats.num_rollbacks += 1;
      } else {
        _stats.num_successful_moves += 1;
      }
    }

    // Revert moves if resulting partition is infeasible
    if (!feasible) {
      tbb::parallel_for(moves.range(), [&](const auto r) {
        for (auto it = r.begin(); it != r.end(); ++it) {
          const auto &move = *it;
          _next_partition[move.u] = _p_graph->block(move.u);
          _p_graph->set_block<false>(move.u, move.from);

          IFSTATS(_stats.rollback_gain += _gains[move.u]);
        }
      });
    } else { // Otherwise, update block weights
      _p_graph->pfor_blocks([&](const BlockID b) {
        _p_graph->set_block_weight(b, _p_graph->block_weight(b) + block_weight_deltas[b]);
      });
    }

    // Update block weights used by LP
    _p_graph->pfor_blocks([&](const BlockID b) { _block_weights[b] = _p_graph->block_weight(b); });

    return feasible;
  }

  void synchronize_state(const NodeID from, const NodeID to) {
    TIMER_BARRIER(_graph->communicator());
    SCOPED_TIMER("Synchronize state");

    struct MoveMessage {
      NodeID local_node;
      BlockID new_block;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<MoveMessage>(
        *_graph,
        from,
        to,

        // We set _next_partition[] to kInvalidBlockID for nodes that were moved
        // during perform_moves()
        [&](const NodeID u) -> bool { return _next_partition[u] == kInvalidBlockID; },

        // Send move to each ghost node adjacent to u
        [&](const NodeID u) -> MoveMessage {
          // perform_moves() marks nodes that were moved locally by setting
          // _next_partition[] to kInvalidBlockID here, we revert this mark
          _next_partition[u] = _p_graph->block(u);

          return {.local_node = u, .new_block = _p_graph->block(u)};
        },

        // move ghost nodes
        [&](const auto recv_buffer, const PEID pe) {
          tbb::parallel_for(
              static_cast<std::size_t>(0), recv_buffer.size(), [&](const std::size_t i) {
                const auto [local_node_on_pe, new_block] = recv_buffer[i];
                const auto global_node =
                    static_cast<GlobalNodeID>(_graph->offset_n(pe) + local_node_on_pe);
                const NodeID local_node = _graph->global_to_local_node(global_node);

                // Otherwise, we should not have gotten this message
                KASSERT(new_block != _p_graph->block(local_node));

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
    KASSERT(u < _graph->n());
    return _p_graph->block(u);
  }

  [[nodiscard]] BlockID cluster(const NodeID u) {
    KASSERT(u < _graph->total_n());
    return _graph->is_owned_node(u) ? _next_partition[u] : _p_graph->block(u);
  }

  void move_node(const NodeID u, const BlockID b) {
    KASSERT(u < _graph->n());
    _next_partition[u] = b;
  }

  [[nodiscard]] BlockWeight initial_cluster_weight(const BlockID b) {
    return _p_graph->block_weight(b);
  }

  [[nodiscard]] BlockWeight cluster_weight(const BlockID b) {
    return __atomic_load_n(&_block_weights[b], __ATOMIC_RELAXED);
  }

  void init_cluster_weight(const BlockID b, const BlockWeight weight) {
    _block_weights[b] = weight;
  }

  [[nodiscard]] BlockWeight max_cluster_weight(const BlockID b) {
    return _p_ctx->max_block_weight(b);
  }

  [[nodiscard]] bool move_cluster_weight(
      const BlockID from, const BlockID to, const BlockWeight delta, const BlockWeight max_weight
  ) {
    if (cluster_weight(to) + delta <= max_weight) {
      __atomic_fetch_add(&_block_weights[to], delta, __ATOMIC_RELAXED);
      __atomic_fetch_sub(&_block_weights[from], delta, __ATOMIC_RELAXED);
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
    return u < _graph->n();
  }

private:
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  bool ASSERT_NEXT_PARTITION_STATE() {
    mpi::barrier(_p_graph->communicator());
    for (const NodeID u : _p_graph->nodes()) {
      if (_p_graph->block(u) != _next_partition[u]) {
        LOG_ERROR << "Invalid _next_partition[] state for node " << u;
        return false;
      }
    }
    mpi::barrier(_p_graph->communicator());
    return true;
  }
#endif

  using Base::_graph;

  const LabelPropagationRefinementContext &_lp_ctx;
  const ParallelContext &_par_ctx;

  DistributedPartitionedGraph *_p_graph = nullptr;
  const PartitionContext *_p_ctx = nullptr;

  StaticArray<BlockID> _next_partition;
  StaticArray<EdgeWeight> _gains;
  StaticArray<BlockWeight> _block_weights;

  lp::RefinerStatistics _stats;
};

//
// Private interface
//

class LPRefinerImplWrapper {
public:
  LPRefinerImplWrapper(const Context &ctx, DistributedPartitionedGraph &p_graph)
      : _csr_impl(std::make_unique<LPRefinerImpl<DistributedCSRGraph>>(ctx, p_graph)),
        _compressed_impl(
            std::make_unique<LPRefinerImpl<DistributedCompressedGraph>>(ctx, p_graph)
        ) {}

  void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    const auto refine = [&](auto &impl, const auto &graph) {
      impl.setup(_memory_context);
      impl.refine(graph, p_graph, p_ctx);
      _memory_context = impl.release();
    };

    p_graph.reified(
        [&](const DistributedCSRGraph &csr_graph) {
          LPRefinerImpl<DistributedCSRGraph> &impl = *_csr_impl;
          refine(impl, csr_graph);
        },
        [&](const DistributedCompressedGraph &compressed_graph) {
          LPRefinerImpl<DistributedCompressedGraph> &impl = *_compressed_impl;
          refine(impl, compressed_graph);
        }
    );
  }

private:
  LPRefinerMemoryContext _memory_context;
  std::unique_ptr<LPRefinerImpl<DistributedCSRGraph>> _csr_impl;
  std::unique_ptr<LPRefinerImpl<DistributedCompressedGraph>> _compressed_impl;
};

//
// Public interface
//

LPRefinerFactory::LPRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
LPRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<LPRefiner>(_ctx, p_graph, p_ctx);
}

LPRefiner::LPRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _impl(std::make_unique<LPRefinerImplWrapper>(ctx, p_graph)),
      _p_graph(p_graph),
      _p_ctx(p_ctx) {}

LPRefiner::~LPRefiner() = default;

void LPRefiner::initialize() {}

bool LPRefiner::refine() {
  _impl->refine(_p_graph, _p_ctx);
  return false;
}

} // namespace kaminpar::dist
