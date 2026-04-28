/*******************************************************************************
 * MultiQueue-based greedy overload balancing.
 *
 * @file:   multi_queue_overload_balancer.cc
 * @author: Daniel Seemaier
 * @date:   29.04.2026
 ******************************************************************************/
#include "kaminpar-shm/refinement/balancer/multi_queue_overload_balancer.h"

#include <limits>
#include <string>
#include <utility>

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/relative_gain.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

} // namespace

MultiQueueOverloadBalancer::MultiQueueOverloadBalancer(const Context &ctx) : _ctx(ctx) {}

MultiQueueOverloadBalancer::~MultiQueueOverloadBalancer() = default;

std::string MultiQueueOverloadBalancer::name() const {
  return "Multi-Queue Overload Balancer";
}

void MultiQueueOverloadBalancer::initialize(const PartitionedGraph &) {
  // Nothing to do.
}

void MultiQueueOverloadBalancer::track_moves(MoveTracker move_tracker) {
  _move_tracker = std::move(move_tracker);
}

bool MultiQueueOverloadBalancer::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Multi-Queue Overload Balancer");
  SCOPED_HEAP_PROFILER("Multi-Queue Overload Balancer");

  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  if (metrics::total_overload(*_p_graph, *_p_ctx) == 0) {
    return false;
  }

  _is_overloaded.resize(p_graph.k());
  _mq.reset(_ctx.parallel.num_threads, /* seed = */ 555);
  _node_target.resize(p_graph.n());
  _node_state.resize(p_graph.n());
  tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID node) {
    _node_state[node] = INACTIVE;
  });

  init_overloaded_blocks();

  reified(*_p_graph, [&]<typename Graph>(const Graph &graph) {
    _gain_cache.emplace<Graph>(_ctx, p_graph.k(), p_graph.k()).initialize(graph, p_graph);
    init_pqs(graph);
  });

  tbb::task_group tg;
  for (int task_id = 0; task_id < _ctx.parallel.num_threads; ++task_id) {
    tg.run([&, task_id] {
      reified(*_p_graph, [&](const auto &graph) { rebalance_worker(graph, task_id); });
    });
  }
  tg.wait();

  return true;
}

template <typename Graph> void MultiQueueOverloadBalancer::init_pqs(const Graph &graph) {
  auto &gain_cache = _gain_cache.get<Graph>();

  [[maybe_unused]] std::atomic<NodeID> num_initial_nodes = 0;
  [[maybe_unused]] std::atomic<NodeID> num_rejected_nodes = 0;

  graph.pfor_nodes([&](const NodeID node) {
    const BlockID from = _p_graph->block(node);
    if (!is_overloaded(from)) {
      return;
    }

    const auto [to, gain] = compute_best_gain(graph, gain_cache, node, from);
    if (to == kInvalidBlockID) {
      IFDBG(++num_rejected_nodes);
      return;
    }

    _node_target[node] = to;
    __atomic_store_n(&_node_state[node], MOVABLE, __ATOMIC_RELAXED);
    insert_node_into_pq(node, to, gain);
    IFDBG(++num_initial_nodes);
  });

  DBG << "Initialized multi-queue overload balancer with " << num_initial_nodes
      << " candidate nodes while skipping " << num_rejected_nodes << " nodes";
}

template <typename Graph>
void MultiQueueOverloadBalancer::rebalance_worker(
    const Graph &graph, [[maybe_unused]] const int task_id
) {
  auto &gain_cache = _gain_cache.get<Graph>();

  while (_num_overloaded_blocks.load(std::memory_order_relaxed) > 0) {
    auto pq = _mq.lock_pop_pq();
    if (!pq) {
      return;
    }

    const NodeID node = pq->peek_id();
    const float expected_gain = pq->peek_key();
    pq->pop();
    _mq.unlock(std::move(pq));

    if (!try_lock_node(node)) {
      continue;
    }

    const BlockID from = _p_graph->block(node);
    if (!is_overloaded(from) || _p_graph->block_weight(from) <= _p_ctx->max_block_weight(from)) {
      deactivate_overloaded_block(from);
      mark_node_moved(node);
      continue;
    }

    const auto [to, actual_gain] = compute_best_gain(graph, gain_cache, node, from);
    if (to == kInvalidBlockID) {
      mark_node_inactive(node);
      continue;
    }

    if (actual_gain < expected_gain) {
      _node_target[node] = to;
      unlock_node(node);
      insert_node_into_pq(node, to, actual_gain);
      continue;
    }

    mark_node_moved(node);
    if (move_node_if_possible(node, from, to)) {
      if (_p_graph->block_weight(from) <= _p_ctx->max_block_weight(from)) {
        deactivate_overloaded_block(from);
      }

      graph.adjacent_nodes(node, [&](const NodeID neighbor) {
        if (!try_lock_node(neighbor)) {
          return;
        }

        const BlockID neighbor_from = _p_graph->block(neighbor);
        if (!is_overloaded(neighbor_from) ||
            _p_graph->block_weight(neighbor_from) <= _p_ctx->max_block_weight(neighbor_from)) {
          deactivate_overloaded_block(neighbor_from);
          unlock_node(neighbor);
          return;
        }

        const auto [neighbor_to, neighbor_gain] =
            compute_best_gain(graph, gain_cache, neighbor, neighbor_from);
        if (neighbor_to == kInvalidBlockID) {
          mark_node_inactive(neighbor);
          return;
        }

        _node_target[neighbor] = neighbor_to;
        unlock_node(neighbor);
        insert_node_into_pq(neighbor, neighbor_to, neighbor_gain);
      });
    } else if (_p_graph->block_weight(from) <= _p_ctx->max_block_weight(from)) {
      deactivate_overloaded_block(from);
    }
  }
}

std::pair<BlockID, float> MultiQueueOverloadBalancer::compute_best_gain(
    const auto &graph, auto &gain_cache, const NodeID node, const BlockID from
) {
  const NodeWeight weight = graph.node_weight(node);
  if (weight == 0) {
    return {kInvalidBlockID, std::numeric_limits<float>::lowest()};
  }

  BlockID best_block = kInvalidBlockID;
  EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
  BlockWeight best_target_weight = _p_graph->block_weight(from) - weight;

  gain_cache.gains(node, from, [&](const BlockID to, auto &&gain_fn) {
    const BlockWeight target_weight = _p_graph->block_weight(to);
    if (target_weight + weight > _p_ctx->max_block_weight(to)) {
      return;
    }

    const EdgeWeight gain = gain_fn();
    if (gain > best_gain || (gain == best_gain && target_weight < best_target_weight)) {
      best_block = to;
      best_gain = gain;
      best_target_weight = target_weight;
    }
  });

  if (best_block == kInvalidBlockID) {
    return {kInvalidBlockID, std::numeric_limits<float>::lowest()};
  }

  return {best_block, compute_relative_gain(best_gain, weight)};
}

void MultiQueueOverloadBalancer::insert_node_into_pq(
    const NodeID node, const BlockID to, const float gain
) {
  KASSERT(to != kInvalidBlockID);

  auto pq = _mq.lock_push_pq();
  pq->push(node, gain);
  _node_target[node] = to;
  _mq.unlock(std::move(pq));
}

void MultiQueueOverloadBalancer::init_overloaded_blocks() {
  [[maybe_unused]] BlockID num_overloaded_blocks = 0;

  _num_overloaded_blocks.store(0, std::memory_order_relaxed);
  for (const BlockID block : _p_graph->blocks()) {
    const std::uint8_t is_overloaded_block = block_overload(block) > 0;
    _is_overloaded[block] = is_overloaded_block;
    _num_overloaded_blocks.fetch_add(is_overloaded_block, std::memory_order_relaxed);
    IFDBG(num_overloaded_blocks += is_overloaded_block);
  }

  DBG << num_overloaded_blocks << " out of " << _p_graph->k() << " blocks are overloaded";
}

bool MultiQueueOverloadBalancer::is_overloaded(const BlockID block) const {
  KASSERT(block < _p_graph->k());

  return __atomic_load_n(&_is_overloaded[block], __ATOMIC_RELAXED);
}

void MultiQueueOverloadBalancer::deactivate_overloaded_block(const BlockID block) {
  KASSERT(block < _p_graph->k());

  std::uint8_t expected = 1u;
  if (__atomic_compare_exchange_n(
          &_is_overloaded[block], &expected, 0u, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
      )) {
    _num_overloaded_blocks.fetch_sub(1, std::memory_order_relaxed);
  }
}

bool MultiQueueOverloadBalancer::try_lock_node(const NodeID node) {
  std::uint8_t expected = MOVABLE;
  return __atomic_load_n(&_node_state[node], __ATOMIC_RELAXED) == MOVABLE &&
         __atomic_compare_exchange_n(
             &_node_state[node], &expected, LOCKED, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED
         );
}

void MultiQueueOverloadBalancer::unlock_node(const NodeID node) {
  KASSERT(__atomic_load_n(&_node_state[node], __ATOMIC_RELAXED) == LOCKED);

  __atomic_store_n(&_node_state[node], MOVABLE, __ATOMIC_RELEASE);
}

void MultiQueueOverloadBalancer::mark_node_moved(const NodeID node) {
  KASSERT(__atomic_load_n(&_node_state[node], __ATOMIC_RELAXED) == LOCKED);

  __atomic_store_n(&_node_state[node], MOVED, __ATOMIC_RELEASE);
}

void MultiQueueOverloadBalancer::mark_node_inactive(const NodeID node) {
  KASSERT(__atomic_load_n(&_node_state[node], __ATOMIC_RELAXED) == LOCKED);

  __atomic_store_n(&_node_state[node], INACTIVE, __ATOMIC_RELEASE);
}

BlockWeight MultiQueueOverloadBalancer::block_overload(const BlockID block) const {
  static_assert(
      std::numeric_limits<BlockWeight>::is_signed,
      "This must be changed when using an unsigned data type for block weights!"
  );

  return std::max<BlockWeight>(0, _p_graph->block_weight(block) - _p_ctx->max_block_weight(block));
}

bool MultiQueueOverloadBalancer::move_node_if_possible(
    const NodeID node, const BlockID from, const BlockID to
) {
  KASSERT(node < _p_graph->n());
  KASSERT(from < _p_graph->k());
  KASSERT(to < _p_graph->k());
  KASSERT(from != to);

  if (_p_graph->move(node, from, to, _p_ctx->max_block_weight(to))) {
    if (_move_tracker != nullptr) {
      _move_tracker(node, from, to);
    }

    return true;
  }

  return false;
}

} // namespace kaminpar::shm
