/*******************************************************************************
 * Greedy balancing algorithms that uses one thread per overloaded block.
 *
 * @file:   overload_balancer.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"

#include <string>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/relative_gain.h"

#include "kaminpar-common/datastructures/dynamic_binary_heap.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

OverloadBalancer::OverloadBalancer(const Context &ctx) : _ctx(ctx) {}

OverloadBalancer::~OverloadBalancer() = default;

std::string OverloadBalancer::name() const {
  return "Overload Balancer";
}

void OverloadBalancer::initialize(const PartitionedGraph &) {
  // Nothing to do: only allocate data if there are any overloaded blocks.
  // Do this in refine() to avoid checking this twice.
}

void OverloadBalancer::track_moves(MoveTracker move_tracker) {
  _move_tracker = move_tracker;
}

bool OverloadBalancer::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Overload Balancer");
  SCOPED_HEAP_PROFILER("Overload Balancer");

  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  if (metrics::total_overload(*_p_graph, *_p_ctx) == 0) {
    return false;
  }

  _moved_nodes.resize(_p_graph->n());
  _pq.init(_p_graph->k());
  _pq_weight.resize(_p_graph->k());

  reified(*_p_graph, [&]<typename Graph>(const Graph &graph) {
    _gain_cache.emplace<Graph>(_ctx, p_graph.k(), p_graph.k()).initialize(graph, p_graph);

    init_pq(graph);
    perform_round(graph);
  });

  return true;
}

template <typename Graph> BlockWeight OverloadBalancer::perform_round(const Graph &graph) {
  // Reset feasible target blocks
  for (auto &blocks : _feasible_target_blocks) {
    blocks.clear();
  }

  tbb::enumerable_thread_specific<BlockWeight> overload_delta;

  auto &gain_cache = _gain_cache.get<Graph>();

  START_TIMER("Main loop");
  SCOPED_HEAP_PROFILER("Main loop");

  tbb::parallel_for<BlockID>(0, _p_graph->k(), [&](const BlockID from) {
    BlockWeight current_overload = block_overload(from);

    if (current_overload > 0 && _feasible_target_blocks.local().empty()) {
      init_feasible_target_blocks();
      DBG << "Block " << from << " with overload: " << current_overload << ": "
          << _feasible_target_blocks.local().size() << " feasible target blocks and "
          << _pq.size(from) << " nodes in PQ: total weight of PQ is " << _pq_weight[from];
    }

    while (current_overload > 0 && !_pq.empty(from)) {
      KASSERT(
          current_overload ==
          std::max<BlockWeight>(0, _p_graph->block_weight(from) - _p_ctx->max_block_weight(from))
      );

      const NodeID u = _pq.peek_max_id(from);
      const NodeWeight u_weight = graph.node_weight(u);
      const float expected_relative_gain = _pq.peek_max_key(from);
      _pq.pop_max(from);
      _pq_weight[from] -= u_weight;
      KASSERT(_moved_nodes[u] == 1);

      auto [to, actual_relative_gain] = compute_best_gain(graph, gain_cache, u, from);

      if (expected_relative_gain <= actual_relative_gain) {
        bool moved_node = false;

        if (to == from) { // Internal node
          moved_node = move_to_random_block(u);
        } else if (move_node_if_possible(u, from, to)) { // Border node
          moved_node = true;
        } else {
          // Move of border node failed (target block no longer viable).
          // In this case, we try again.
        }

        // If node was moved, update overload of its original block.
        if (moved_node) {
          const BlockWeight delta = std::min(current_overload, u_weight);
          current_overload -= delta;
          overload_delta.local() += delta;

          // Try to add neighbors of moved node to PQ
          graph.adjacent_nodes(u, [&](const NodeID v) {
            if (_moved_nodes[v] == 0 && _p_graph->block(v) == from) {
              const auto [to, rel_gain] = compute_best_gain(graph, gain_cache, v, from);
              add_to_pq(from, v, _p_graph->node_weight(v), rel_gain);
              _moved_nodes[v] = 1;
            }
          });
        } else if (to != from) {
          // Only re-insert nodes that we tried to move to adjacent blocks
          add_to_pq(from, u, u_weight, actual_relative_gain);
        }
      } else {
        // If the gain of the node worsened (this only happens if its original target block is no
        // longer viable), re-insert the node with its new gain value.
        add_to_pq(from, u, _p_graph->node_weight(u), actual_relative_gain);
      }
    }

    KASSERT(
        current_overload ==
        std::max<BlockWeight>(0, _p_graph->block_weight(from) - _p_ctx->max_block_weight(from))
    );
  });

  STOP_TIMER();

  return overload_delta.combine(std::plus{});
}

template <typename Graph> void OverloadBalancer::init_pq(const Graph &graph) {
  SCOPED_TIMER("Initialize balancer PQ");
  SCOPED_HEAP_PROFILER("Initialize balancer PQ");

  const BlockID k = _p_graph->k();
  auto &gain_cache = _gain_cache.get<Graph>();

  using PQs = std::vector<DynamicBinaryMinHeap<NodeID, float, ScalableVector>>;
  tbb::enumerable_thread_specific<PQs> local_pq{[&] {
    return PQs(k);
  }};

  using PQWeights = std::vector<NodeWeight>;
  tbb::enumerable_thread_specific<PQWeights> local_pq_weight{[&] {
    return PQWeights(k);
  }};

  // build thread-local PQs: one PQ for each thread and block, each PQ for block
  // b has at most roughly |overload[b]| weight
  START_TIMER("Thread-local");
  tbb::parallel_for<NodeID>(0, graph.n(), [&](const NodeID u) {
    auto &pq = local_pq.local();
    auto &pq_weight = local_pq_weight.local();

    const BlockID b = _p_graph->block(u);
    const BlockWeight overload = block_overload(b);

    if (overload > 0) { // node in overloaded block
      const auto [max_gainer, rel_gain] = compute_best_gain(graph, gain_cache, u, b);
      const bool need_more_nodes = (pq_weight[b] < overload);
      if (need_more_nodes || pq[b].empty() || rel_gain > pq[b].peek_key()) {
        if (!need_more_nodes) {
          const NodeWeight u_weight = _p_graph->node_weight(u);
          const NodeWeight min_weight = _p_graph->node_weight(pq[b].peek_id());
          if (pq_weight[b] + u_weight - min_weight >= overload) {
            pq[b].pop();
          }
        }
        pq[b].push(u, rel_gain);
        _moved_nodes[u] = 1;
      }
    }
  });
  STOP_TIMER();

  // build global PQ: one PQ per block, block-level parallelism
  _pq.clear();

  START_TIMER("Merge thread-local PQs");
  tbb::parallel_for(static_cast<BlockID>(0), k, [&](const BlockID b) {
    _pq_weight[b] = 0;

    for (auto &pq : local_pq) {
      for (const auto &[u, rel_gain] : pq[b].elements()) {
        add_to_pq(b, u, _p_graph->node_weight(u), rel_gain);
      }
    }
  });
  STOP_TIMER();
}

std::pair<BlockID, float> OverloadBalancer::compute_best_gain(
    const auto &graph, auto &gain_cache, const NodeID node, const BlockID from
) {
  const NodeWeight weight = graph.node_weight(node);
  BlockID best_block = kInvalidBlockID;
  EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();

  gain_cache.gains(node, from, [&](const BlockID to, auto &&gain_fn) {
    if (_p_graph->block_weight(to) + weight <= _p_ctx->max_block_weight(to)) {
      const EdgeWeight gain = gain_fn();
      if (gain >= best_gain) {
        best_block = to;
        best_gain = gain;
      }
    }
  });

  return std::make_pair(best_block, compute_relative_gain(best_gain, weight));
}

bool OverloadBalancer::add_to_pq(
    const BlockID block, const NodeID node, const NodeWeight weight, const float rel_gain
) {
  KASSERT(weight == _p_graph->node_weight(node));
  KASSERT(block == _p_graph->block(node));

  if (_pq_weight[block] < block_overload(block) || _pq.empty(block) ||
      rel_gain > _pq.peek_min_key(block)) {
    _pq.push(block, node, rel_gain);
    _pq_weight[block] += weight;

    if (rel_gain > _pq.peek_min_key(block)) {
      const NodeID min_node = _pq.peek_min_id(block);
      const NodeWeight min_weight = _p_graph->node_weight(min_node);
      if (_pq_weight[block] - min_weight >= block_overload(block)) {
        _pq.pop_min(block);
        _pq_weight[block] -= min_weight;
      }
    }

    return true;
  }

  return false;
}

bool OverloadBalancer::move_node_if_possible(
    const NodeID node, const BlockID from, const BlockID to
) {
  if (_p_graph->move(node, from, to, _p_ctx->max_block_weight(to))) {
    if (_move_tracker != nullptr) {
      _move_tracker(node, from, to);
    }

    return true;
  }

  return false;
}

bool OverloadBalancer::move_to_random_block(const NodeID node) {
  auto &feasible_target_blocks = _feasible_target_blocks.local();
  const BlockID from = _p_graph->block(node);

  while (!feasible_target_blocks.empty()) {
    // Get random block from feasible block list
    const std::size_t n = feasible_target_blocks.size();
    const std::size_t i = Random::instance().random_index(0, n);
    const BlockID to = feasible_target_blocks[i];
    KASSERT(from != to);

    // Try to move node to that block, if possible, operation succeeded
    if (move_node_if_possible(node, from, to)) {
      return true;
    }

    // Loop terminated without return, hence moving `node` to `to` failed.
    // In this case, we no longer consider b to be a feasible target block and remove it from the
    // list.
    std::swap(feasible_target_blocks[i], feasible_target_blocks.back());
    feasible_target_blocks.pop_back();
  }

  // There are no more feasible target blocks -> operation failed
  return false;
}

void OverloadBalancer::init_feasible_target_blocks() {
  auto &blocks = _feasible_target_blocks.local();
  blocks.clear();

  for (const BlockID b : _p_graph->blocks()) {
    if (_p_graph->block_weight(b) < _p_ctx->perfectly_balanced_block_weight(b)) {
      blocks.push_back(b);
    }
  }
}

BlockWeight OverloadBalancer::block_overload(const BlockID block) const {
  static_assert(
      std::numeric_limits<BlockWeight>::is_signed,
      "This must be changed when using an unsigned data type for "
      "block weights!"
  );

  return std::max<BlockWeight>(0, _p_graph->block_weight(block) - _p_ctx->max_block_weight(block));
}

} // namespace kaminpar::shm
