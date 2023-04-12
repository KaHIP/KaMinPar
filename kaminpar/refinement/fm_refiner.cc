/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#include "kaminpar/refinement/fm_refiner.h"

#include <cmath>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"

#include "common/datastructures/marker.h"
#include "common/parallel/atomic.h"

namespace kaminpar::shm {
namespace {
struct StoppingPolicy {
  StoppingPolicy(const double alpha) : _factor(alpha / 2.0 - 0.25) {}

  void init(const NodeID n) {
    _beta = std::log(n);
  }

  [[nodiscard]] bool should_stop() const {
    return (_num_steps > _beta) &&
           ((_Mk == 0) || (_num_steps >= (_variance / (_Mk * _Mk)) * _factor));
  }

  void reset() {
    _num_steps = 0;
    _variance = 0.0;
  }

  void update(const EdgeWeight gain) {
    ++_num_steps;

    // See Knuth TAOCP vol 2, 3rd edition, page 232 or
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    if (_num_steps == 1) {
      _MkMinus1 = static_cast<double>(gain);
      _Mk = _MkMinus1;
      _SkMinus1 = 0.0;
    } else {
      _Mk = _MkMinus1 + (gain - _MkMinus1) / _num_steps;
      _Sk = _SkMinus1 + (gain - _MkMinus1) * (gain - _Mk);
      _variance = _Sk / (_num_steps - 1.0);

      // Prepare for next iteration
      _MkMinus1 = _Mk;
      _SkMinus1 = _Sk;
    }
  }

private:
  double _factor;

  double _beta = 0.0;
  std::size_t _num_steps = 0;
  double _variance = 0.0;
  double _Mk = 0.0;
  double _MkMinus1 = 0.0;
  double _Sk = 0.0;
  double _SkMinus1 = 0.0;
};

struct Move {
  NodeID node;
  BlockID from;
  BlockID to;
};
} // namespace

FMRefiner::FMRefiner(const Context &ctx)
    : _fm_ctx(&ctx.refinement.kway_fm),
      _locked(ctx.partition.n),
      _pq_ets([&] { return BinaryMinHeap<EdgeWeight>(ctx.partition.n); }),
      _marker_ets([&] { return Marker(ctx.partition.n); }),
      _target_blocks(ctx.partition.n) {}

void FMRefiner::initialize(const Graph &graph) {
  ((void)graph);
}

bool FMRefiner::run_localized_refinement() {
  auto &pq = _pq_ets.local();
  auto &marker = _marker_ets.local();

  pq.clear();

  DeltaPartitionedGraph d_graph(_p_graph);
  StoppingPolicy stopping_policy(_fm_ctx->alpha);
  std::vector<NodeID> touched_nodes;

  poll_border_nodes(_fm_ctx->num_seed_nodes, [&](const NodeID u) {
    const auto [block, gain] = _gain_cache.best_gain(u, _p_ctx->block_weights);
    pq.push(u, gain);
    _target_blocks[u] = block;
    touched_nodes.push_back(u);
  });

  EdgeWeight total_gain = 0;
  std::vector<Move> moves;

  while (!pq.empty() && !stopping_policy.should_stop()) {
    const NodeID node = pq.peek_id();
    const EdgeWeight gain = pq.peek_key();
    const BlockID to = _target_blocks[node];
    const NodeWeight node_weight = _p_graph->node_weight(node);
    pq.pop();

    // If the actual gain is worse than the PQ gain, recompute
    if (gain > _gain_cache.gain_to(node, to)) {
      const auto [block, gain] =
          _gain_cache.best_gain(node, _p_ctx->block_weights);
      pq.push(node, gain);
      _target_blocks[node] = block;

      // ... and try again
      continue;
    }

    // Otherwise, try to move the node
    if (d_graph.block_weight(to) + node_weight <=
        _p_ctx->block_weights.max(to)) {
      total_gain += gain;
      moves.push_back({node, d_graph.block(node), to});
      d_graph.set_block(node, to);
      stopping_policy.update(gain);

      if (total_gain > 0) {
        for (const auto &[u, b] : d_graph.delta()) {
          _gain_cache.move_node(u, _p_graph->block(u), b);
          _p_graph->set_block(u, b);
        }
        moves.clear();
      }

      for (const auto &[e, v] : _p_graph->neighbors(node)) {
        if (!lock_node(v)) {
          continue;
        }

        const auto [block, gain] =
            _gain_cache.best_gain(v, _p_ctx->block_weights);
        if (pq.contains(v)) {
          pq.change_priority(v, gain);
        } else {
          touched_nodes.push_back(v);
          pq.push(v, gain);
        }
        _target_blocks[v] = block;
      }
    } else {
    }
  }

  for (const NodeID u : touched_nodes) {
    unlock_node(u);
  }

  return total_gain > 0;
}

bool FMRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  p_graph.pfor_nodes([&](NodeID u) { _locked[u] = 0; });
  _gain_cache.reinit(_p_graph);
  init_border_nodes();

  std::atomic<std::uint8_t> found_improvement = 0;
  tbb::parallel_for<int>(0, tbb::this_task_arena::max_concurrency(), [&](int) {
    while (has_border_nodes()) {
      found_improvement |= run_localized_refinement();
    }
  });

  return found_improvement;
}

void FMRefiner::init_border_nodes() {
  _border_nodes.clear();
  _p_graph->pfor_nodes([&](const NodeID u) {
    if (_gain_cache.is_border_node(u)) {
      _border_nodes.push_back(u);
    }
    _locked[u] = 0;
  });
}

bool FMRefiner::has_border_nodes() const {
  return _next_border_node < _border_nodes.size();
}

bool FMRefiner::lock_node(const NodeID u) {
  std::uint8_t free = 0;
  return __atomic_compare_exchange_n(
      &_locked[u], &free, 1, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
  );
}

void FMRefiner::unlock_node(const NodeID u) {
  _locked[u] = 0;
  if (_gain_cache.is_border_node(u)) {
    _border_nodes.push_back(u);
  }
}
} // namespace kaminpar::shm
