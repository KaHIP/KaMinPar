/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 * @brief:  Distributed FM refiner.
 ******************************************************************************/
#include "dkaminpar/refinement/fm_refiner.h"

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/algorithms/independent_set.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/graphutils/bfs_extractor.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/refinement/move_conflict_resolver.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/rating_map.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
namespace {
struct AdaptiveStoppingPolicy {
  AdaptiveStoppingPolicy(const GlobalNodeID n) : _beta(std::log(n)) {}

  [[nodiscard]] bool should_stop(const double alpha) const {
    const double factor = alpha / 2.0 - 0.25;
    return (_num_steps > _beta) &&
           ((_mk == 0.0) || (_num_steps >= (_variance / (_mk * _mk)) * factor));
  }

  void reset() {
    _num_steps = 0;
    _variance = 0.0;
  }

  void update(const EdgeWeight gain) {
    ++_num_steps;

    // see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    if (_num_steps == 1) {
      _mk_minus_one = 1.0 * gain;
      _mk = _mk_minus_one;
      _sk_minus_one = 0.0;
    } else {
      _mk = _mk_minus_one + (gain - _mk_minus_one) / _num_steps;
      _sk = _sk_minus_one + (gain - _mk_minus_one) * (gain - _mk);
      _variance = _sk / (_num_steps - 1.0);
      _mk_minus_one = _mk;
      _sk_minus_one = _sk;
    }
  }

private:
  double _beta{0.0};
  std::size_t _num_steps{0};
  double _variance{0.0};
  double _mk{0.0};
  double _mk_minus_one{0.0};
  double _sk{0.0};
  double _sk_minus_one{0.0};
};

class ThreadLocalFMRefiner {
  SET_DEBUG(false);

public:
  struct Move {
    Move(const NodeID node, const EdgeWeight gain, const BlockID block)
        : node(node),
          gain(gain),
          block(block) {}

    NodeID node;
    EdgeWeight gain;
    BlockID block;
  };

  ThreadLocalFMRefiner(
      const DistributedPartitionedGraph &global_p_graph,
      const shm::PartitionedGraph &p_graph,
      const FMRefinementContext &fm_ctx,
      const PartitionContext &p_ctx
  )
      : _global_p_graph(global_p_graph),
        _p_graph(p_graph),
        _fm_ctx(fm_ctx),
        _p_ctx(p_ctx),
        _rating_map(_p_ctx.k),
        _stopping_policy(_p_ctx.graph->global_n),
        _pq(_p_graph.n()),
        _marker(_p_graph.n()) {}

  std::vector<Move> refine(const NodeID start_node) {
    initialize(start_node);

    EdgeWeight best_total_gain = 0;
    EdgeWeight current_total_gain = 0;
    std::size_t rollback_index = 0;
    std::vector<Move> moves;

    while (!_pq.empty() && !_stopping_policy.should_stop(_fm_ctx.alpha)) {
      // Retrieve next node from PQ
      const NodeID u = _pq.peek_id();
      const NodeWeight weight = _p_graph.node_weight(u);
      const BlockID from = block(u);
      const auto [gain, to] = find_best_target_block<false>(u);
      _pq.pop();

      // Only perform move if target block can take u without becoming
      // overloaded
      const bool feasible =
          to != from && _global_p_graph.block_weight(to) + _block_weight_deltas[to] + weight <=
                            _p_ctx.graph->max_block_weight(to);

      if (feasible) {
        /*DBG << "Feasible move with gain " << gain << ": " << u << " > " <<
        from << " --> " << to; if (u == 226) { DBG << "Neighbors: (n= " <<
        _p_graph.n() << ")"; for (const auto& [e, v]: _p_graph.neighbors(u)) {
                DBG << V(v) << " with edge weight " << _p_graph.edge_weight(e)
        << " block " << block(v);
            }
        }*/

        // Move u to its target block
        set_block(u, to);
        _block_weight_deltas[from] -= weight;
        _block_weight_deltas[to] += weight;
        moves.emplace_back(u, gain, from);
        update_pq_after_move(u, from, to);
        current_total_gain += gain;
        _stopping_policy.update(gain);

        if (current_total_gain > best_total_gain) {
          best_total_gain = current_total_gain;
          rollback_index = moves.size();
          _stopping_policy.reset();
        }
      } else {
        // Target block became too full since PQ initialization
        // --> retry with new target block
        insert_node_into_pq<false>(u);
      }
    }

    DBG << "Accepted gain: " << best_total_gain;

    // Rollback to best cut found
    while (moves.size() > rollback_index) {
      const auto [node, gain, to] = moves.back();
      const BlockID from = block(node);
      const NodeWeight weight = _p_graph.node_weight(node);

      _block_weight_deltas[from] -= weight;
      _block_weight_deltas[to] += weight;
      set_block(node, to);

      moves.pop_back();
    }

    // For the moves that we keep, store the new block rather than the old one
    for (auto &[u, gain, node_block] : moves) {
      node_block = block(u);
    }

    return moves;
  }

private:
  void initialize(const NodeID start_node) {
    // Clear data structures from previous run
    _marker.reset();
    _pq.clear();
    _stopping_policy.reset();
    _block_weight_deltas.clear();
    _partition_delta.clear();

    // Fill PQ with start node and its neighbors
    insert_node_into_pq<true>(start_node);
    for (const auto [e, neighbor] : _p_graph.neighbors(start_node)) {
      if (!is_pseudo_node_block(neighbor)) {
        insert_node_into_pq<true>(neighbor);
      }
    }
  }

  void update_pq_after_move(const NodeID u, const BlockID from, const BlockID to) {
    for (const auto [e, v] : _p_graph.neighbors(u)) {
      KASSERT(v < _p_graph.n());

      if (is_pseudo_node_block(v)) {
        continue;
      }

      const BlockID v_block = block(v);
      if (v_block == from || v_block == to) {
        const auto [best_gain, best_target_block] = find_best_target_block<false>(v);
        if (_pq.contains(v)) {
          _pq.change_priority(v, best_gain);
        } else if (!_marker.get(v)) {
          insert_node_into_pq<false>(v);
          _marker.set(v);
        }
      }
    }
  }

  template <bool initialization> void insert_node_into_pq(const NodeID u) {
    KASSERT(u < _p_graph.n());

    const auto [best_gain, best_target_block] = find_best_target_block<initialization>(u);
    if (block(u) != best_target_block) {
      _pq.push(u, best_gain);
      if (initialization) {
        _marker.set(u);
      }
    }
  }

  template <bool initialization>
  std::pair<EdgeWeight, BlockID> find_best_target_block(const NodeID u) {
    const BlockID u_block = block(u);

    auto action = [&](auto &map) {
      EdgeWeight internal_degree = 0;
      for (const auto [e, v] : _p_graph.neighbors(u)) {
        KASSERT(v < _p_graph.n(), V(_p_graph.n()) << V(u));

        const BlockID v_block = block(v);
        const EdgeWeight e_weight = _p_graph.edge_weight(e);

        if (v_block != u_block) {
          map[v_block] += e_weight;
        } else {
          internal_degree += e_weight;
        }
      }

      EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
      BlockID best_target_block = u_block;

      const NodeWeight u_weight = _p_graph.node_weight(u);

      for (const auto [current_target_block, current_gain] : map.entries()) {
        // compute weight of block if we were to move the node
        BlockWeight block_weight_prime =
            _global_p_graph.block_weight(current_target_block) + u_weight;
        if (!initialization) {
          block_weight_prime += _block_weight_deltas[current_target_block];
        }
        const bool feasible =
            block_weight_prime <= _p_ctx.graph->max_block_weight(current_target_block);

        // accept as better block if gain is larger
        // if gain is equal, flip a coin
        if (feasible &&
            (current_gain > best_gain || (current_gain == best_gain && _rand.random_bool()))) {
          best_gain = current_gain;
          best_target_block = current_target_block;
        }
      }

      // subtract internal degree to get the actual gain value of this move
      best_gain -= internal_degree; // overflow OK, value unused if still set to min

      map.clear(); // clear for next node
      return std::make_pair(best_gain, best_target_block);
    };

    _rating_map.update_upper_bound_size(std::min<BlockID>(_p_ctx.k, _p_graph.degree(u)));
    return _rating_map.run_with_map(action, action);
  }

  BlockID block(const NodeID node) {
    if (_partition_delta.find(node) != _partition_delta.end()) {
      return _partition_delta[node];
    } else {
      return _p_graph.block(node);
    }
  }

  void set_block(const NodeID node, const BlockID block) {
    _partition_delta[node] = block;
  }

  bool is_pseudo_node_block(const NodeID node) {
    return node > _p_graph.n() - _p_ctx.k;
  }

  /*
   * Initialized by constructor
   */
  const DistributedPartitionedGraph &_global_p_graph;
  const shm::PartitionedGraph &_p_graph;
  const FMRefinementContext &_fm_ctx;
  const PartitionContext &_p_ctx;
  RatingMap<EdgeWeight, NodeID> _rating_map;
  AdaptiveStoppingPolicy _stopping_policy;
  BinaryMaxHeap<EdgeWeight> _pq;
  Marker<> _marker;

  /*
   * Initialized here
   */
  Random &_rand{Random::instance()};
  std::unordered_map<BlockID, BlockWeight> _block_weight_deltas;
  std::unordered_map<NodeID, BlockID> _partition_delta;
};
} // namespace

FMRefiner::FMRefiner(const Context &ctx) : _fm_ctx(ctx.refinement.fm) {}

void FMRefiner::initialize(const DistributedGraph &) {}

void FMRefiner::refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("FM");
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  for (std::size_t global_round = 0; global_round < 5; ++global_round) {
    // Find independent set of border nodes
    START_TIMER("Call find_independent_border_set");
    const auto seed_nodes = graph::find_independent_border_set(*_p_graph, global_round);
    STOP_TIMER();

    mpi::barrier(_p_graph->communicator());

    // Run BFS
    START_TIMER("Call BfsExtractor");
    graph::BfsExtractor bfs_extractor(_p_graph->graph());
    bfs_extractor.initialize(*_p_graph);
    bfs_extractor.set_max_hops(1);
    bfs_extractor.set_max_radius(2);
    auto extraction_result = bfs_extractor.extract(seed_nodes);
    STOP_TIMER();

    mpi::barrier(_p_graph->communicator());

    START_TIMER("Build reverse node mapping");
    growt::StaticGhostNodeMapping reverse_node_mapping(extraction_result.node_mapping.size());
    tbb::parallel_for<std::size_t>(
        0,
        extraction_result.node_mapping.size(),
        [&](const std::size_t i) {
          reverse_node_mapping.insert(extraction_result.node_mapping[i] + 1, i);
        }
    );
    STOP_TIMER();

    mpi::barrier(_p_graph->communicator());

    // Run FM
    tbb::enumerable_thread_specific<ThreadLocalFMRefiner> fm_refiner_ets{[&] {
      return ThreadLocalFMRefiner(*_p_graph, *extraction_result.p_graph, _fm_ctx, *_p_ctx);
    }};

    for (std::size_t local_round = 0; local_round < 5; ++local_round) {
      DBG << "Starting FM round " << local_round;

      START_TIMER("Local FM");
      tbb::concurrent_vector<GlobalMove> local_move_buffer;
      tbb::parallel_for<std::size_t>(0, seed_nodes.size(), [&](const std::size_t i) {
        const NodeID local_seed_node = seed_nodes[i];
        const GlobalNodeID global_seed_node = _p_graph->local_to_global_node(local_seed_node);

        KASSERT(reverse_node_mapping.find(global_seed_node + 1) != reverse_node_mapping.end());
        const NodeID seed_node = (*reverse_node_mapping.find(global_seed_node + 1)).second;

        auto &fm_refiner = fm_refiner_ets.local();
        auto moves = fm_refiner.refine(seed_node);

        const NodeID group = local_seed_node;
        const auto &p_graph = extraction_result.p_graph;
        const auto &node_mapping = extraction_result.node_mapping;
        for (const auto &[node, gain, to] : moves) {
          KASSERT(node < node_mapping.size());
          local_move_buffer.push_back(
              {node_mapping[node],
               group,
               static_cast<NodeWeight>(p_graph->node_weight(node)),
               gain,
               p_graph->block(node),
               to}
          );
        }
      });
      STOP_TIMER();

      mpi::barrier(_p_graph->communicator());

      // Resolve global move conflicts
      START_TIMER("Move conflict resolution");
      std::vector<GlobalMove> local_move_buffer_cpy(
          local_move_buffer.begin(), local_move_buffer.end()
      );
      auto global_move_buffer =
          broadcast_and_resolve_global_moves(local_move_buffer_cpy, _p_graph->communicator());
      // auto global_move_buffer = local_move_buffer_cpy;
      STOP_TIMER();

      mpi::barrier(_p_graph->communicator());

      // Apply moves to global partition and extract graph
      START_TIMER("Apply moves");
      for (const auto &[node, group, weight, gain, from, to] : global_move_buffer) {
        if (node == kInvalidGlobalNodeID) {
          continue; // Move conflicts with a better move
        }

        // Apply move to distributed graph
        if (_p_graph->contains_global_node(node)) {
          const NodeID local_node = _p_graph->global_to_local_node(node);
          _p_graph->set_block(local_node, to);
        } else {
          _p_graph->set_block_weight(from, _p_graph->block_weight(from) - weight);
          _p_graph->set_block_weight(to, _p_graph->block_weight(to) + weight);
        }

        // Apply move to local graph (if contained in local graph)
        auto it = reverse_node_mapping.find(node + 1);
        if (it != reverse_node_mapping.end()) {
          const NodeID extracted_node = (*it).second;
          extraction_result.p_graph->set_block<false>(extracted_node, to);
        }
      }
      STOP_TIMER();

      mpi::barrier(_p_graph->communicator());
    }
  }
}
} // namespace kaminpar::dist
