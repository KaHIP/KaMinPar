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

#include <tbb/enumerable_thread_specific.h>
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

MultiQueueOverloadBalancer::AccessToken::AccessToken(const int seed, const std::size_t num_pqs)
    : dist(0, num_pqs - 1) {
  rng.seed(seed);
}

std::size_t MultiQueueOverloadBalancer::AccessToken::pick_random_pq() {
  return dist(rng);
}

std::array<std::size_t, 2> MultiQueueOverloadBalancer::AccessToken::pick_two_random_pqs() {
  std::array<std::size_t, 2> pqs{pick_random_pq(), pick_random_pq()};
  while (pqs[0] == pqs[1]) {
    pqs[1] = pick_random_pq();
  }
  return pqs;
}

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
  _node_target.resize(p_graph.n());
  _node_pq.resize(p_graph.n());
  _pq_handles.resize(p_graph.n());
  _node_state.resize(p_graph.n());
  tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID node) {
    _node_state[node] = INACTIVE;
    _node_pq[node] = std::numeric_limits<std::size_t>::max();
    _pq_handles[node] = PQ::kInvalidID;
  });

  const std::size_t num_pqs = std::max(2, 2 * _ctx.parallel.num_threads);
  _pqs.clear();
  _pqs.reserve(num_pqs);
  for (std::size_t i = 0; i < num_pqs; ++i) {
    _pqs.emplace_back(p_graph.n(), _pq_handles.data());
  }
  _pq_locks.assign(num_pqs, 0);
  _pq_top_keys.assign(num_pqs, std::numeric_limits<float>::lowest());

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

  clear_pqs();

  return true;
}

template <typename Graph> void MultiQueueOverloadBalancer::init_pqs(const Graph &graph) {
  auto &gain_cache = _gain_cache.get<Graph>();
  std::atomic<int> seed{555};
  tbb::enumerable_thread_specific<AccessToken> tokens([&] {
    return AccessToken(seed.fetch_add(1, std::memory_order_relaxed), _pqs.size());
  });

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
    insert_node_into_pq(node, to, gain, tokens.local());
    IFDBG(++num_initial_nodes);
  });

  DBG << "Initialized multi-queue overload balancer with " << num_initial_nodes
      << " candidate nodes while skipping " << num_rejected_nodes << " nodes";
}

template <typename Graph>
void MultiQueueOverloadBalancer::rebalance_worker(const Graph &graph, const int task_id) {
  auto &gain_cache = _gain_cache.get<Graph>();
  AccessToken token(graph.n() + task_id, _pqs.size());

  while (_num_overloaded_blocks.load(std::memory_order_relaxed) > 0) {
    Move move;
    if (!find_next_move(graph, gain_cache, token, move)) {
      return;
    }

    mark_node_moved(move.node);

    if (!is_overloaded(move.from) ||
        _p_graph->block_weight(move.from) <= _p_ctx->max_block_weight(move.from)) {
      deactivate_overloaded_block(move.from);
      continue;
    }

    if (move_node_if_possible(move.node, move.from, move.to)) {
      if (_p_graph->block_weight(move.from) <= _p_ctx->max_block_weight(move.from)) {
        deactivate_overloaded_block(move.from);
      }

      update_neighbors(graph, gain_cache, move);
    } else if (_p_graph->block_weight(move.from) <= _p_ctx->max_block_weight(move.from)) {
      deactivate_overloaded_block(move.from);
    }
  }
}

template <typename Graph>
bool MultiQueueOverloadBalancer::find_next_move(
    const Graph &graph, auto &gain_cache, AccessToken &token, Move &move
) {
  static constexpr int kNumPopAttempts = 32;

  for (int attempt = 0; attempt < kNumPopAttempts; ++attempt) {
    const auto [first, second] = token.pick_two_random_pqs();

    const std::size_t first_lock = std::min(first, second);
    const std::size_t second_lock = std::max(first, second);
    if (!try_lock_pq(first_lock)) {
      continue;
    }
    if (!try_lock_pq(second_lock)) {
      unlock_pq(first_lock);
      continue;
    }

    const bool first_empty = _pqs[first].empty();
    const bool second_empty = _pqs[second].empty();
    if (first_empty && second_empty) {
      unlock_pq(second_lock);
      unlock_pq(first_lock);
      continue;
    }

    const std::size_t pq =
        (first_empty || (!second_empty && _pqs[first].peek_key() < _pqs[second].peek_key()))
            ? second
            : first;
    const std::size_t other = pq == first ? second : first;
    unlock_pq(other);

    if (_pqs[pq].empty()) {
      unlock_pq(pq);
      continue;
    }
    if (locked_extract_candidate(pq, graph, gain_cache, move)) {
      return true;
    }

    attempt = 0;
  }

  while (true) {
    float best_key = std::numeric_limits<float>::lowest();
    std::size_t best_pq = std::numeric_limits<std::size_t>::max();

    for (std::size_t pq = 0; pq < _pqs.size(); ++pq) {
      if (!try_lock_pq(pq)) {
        continue;
      }

      if (!_pqs[pq].empty() && _pqs[pq].peek_key() > best_key) {
        if (best_pq != std::numeric_limits<std::size_t>::max()) {
          unlock_pq(best_pq);
        }

        best_key = _pqs[pq].peek_key();
        best_pq = pq;
      } else {
        unlock_pq(pq);
      }
    }

    if (best_pq == std::numeric_limits<std::size_t>::max()) {
      bool any_locked = false;
      for (std::size_t pq = 0; pq < _pqs.size(); ++pq) {
        any_locked |= __atomic_load_n(&_pq_locks[pq], __ATOMIC_RELAXED) != 0;
      }
      if (any_locked) {
        continue;
      } else {
        return false;
      }
    }

    if (locked_extract_candidate(best_pq, graph, gain_cache, move)) {
      return true;
    }
  }
}

template <typename Graph>
bool MultiQueueOverloadBalancer::locked_extract_candidate(
    const std::size_t pq_id, const Graph &graph, auto &gain_cache, Move &move
) {
  auto &pq = _pqs[pq_id];
  const NodeID node = pq.peek_id();
  const float expected_gain = pq.peek_key();

  if (!try_lock_node(node)) {
    unlock_pq(pq_id);
    return false;
  }

  const BlockID from = _p_graph->block(node);
  if (!is_overloaded(from) || _p_graph->block_weight(from) <= _p_ctx->max_block_weight(from)) {
    pq.pop();
    deactivate_overloaded_block(from);
    mark_node_moved(node);
    unlock_pq(pq_id);
    return false;
  }

  const auto [to, actual_gain] = compute_best_gain(graph, gain_cache, node, from);
  if (to != kInvalidBlockID && actual_gain >= expected_gain) {
    pq.pop();
    move.node = node;
    move.from = from;
    move.to = to;
    move.gain = actual_gain;
    unlock_pq(pq_id);
    return true;
  }

  if (to != kInvalidBlockID) {
    _node_target[node] = to;
    pq.change_priority(node, actual_gain);
    unlock_node(node);
  } else {
    pq.pop();
    mark_node_inactive(node);
  }

  unlock_pq(pq_id);
  return false;
}

template <typename Graph>
void MultiQueueOverloadBalancer::update_neighbors(
    const Graph &graph, auto &gain_cache, const Move move
) {
  graph.adjacent_nodes(move.node, [&](const NodeID neighbor) {
    if (neighbor == move.node || !try_lock_node(neighbor)) {
      return;
    }

    const BlockID neighbor_from = _p_graph->block(neighbor);
    if (!is_overloaded(neighbor_from) ||
        _p_graph->block_weight(neighbor_from) <= _p_ctx->max_block_weight(neighbor_from)) {
      deactivate_overloaded_block(neighbor_from);
      _node_target[neighbor] = kInvalidBlockID;
      remove_node_from_pq(neighbor);
      unlock_node(neighbor);
      return;
    }

    const auto [neighbor_to, neighbor_gain] =
        compute_best_gain(graph, gain_cache, neighbor, neighbor_from);
    _node_target[neighbor] = neighbor_to;

    if (neighbor_to == kInvalidBlockID) {
      remove_node_from_pq(neighbor);
    } else {
      update_node_in_pq(neighbor, neighbor_gain);
    }

    unlock_node(neighbor);
  });
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
    const NodeID node, const BlockID to, const float gain, AccessToken &token
) {
  KASSERT(to != kInvalidBlockID);
  KASSERT(__atomic_load_n(&_node_state[node], __ATOMIC_RELAXED) == MOVABLE);

  const std::size_t pq = token.pick_random_pq();
  lock_pq(pq);
  _pqs[pq].push(node, gain);
  _node_target[node] = to;
  _node_pq[node] = pq;
  unlock_pq(pq);
}

void MultiQueueOverloadBalancer::update_node_in_pq(const NodeID node, const float gain) {
  const std::size_t pq = _node_pq[node];
  KASSERT(pq < _pqs.size());

  lock_pq(pq);
  if (_pqs[pq].contains(node)) {
    _pqs[pq].change_priority(node, gain);
  }
  unlock_pq(pq);
}

void MultiQueueOverloadBalancer::remove_node_from_pq(const NodeID node) {
  const std::size_t pq = _node_pq[node];
  KASSERT(pq < _pqs.size());

  lock_pq(pq);
  if (_pqs[pq].contains(node)) {
    _pqs[pq].remove(node);
  }
  unlock_pq(pq);
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

bool MultiQueueOverloadBalancer::try_lock_pq(const std::size_t pq) {
  KASSERT(pq < _pq_locks.size());

  std::uint8_t expected = 0;
  return __atomic_compare_exchange_n(
      &_pq_locks[pq], &expected, 1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED
  );
}

void MultiQueueOverloadBalancer::lock_pq(const std::size_t pq) {
  while (!try_lock_pq(pq)) {
  }
}

void MultiQueueOverloadBalancer::unlock_pq(const std::size_t pq) {
  KASSERT(pq < _pq_locks.size());
  KASSERT(__atomic_load_n(&_pq_locks[pq], __ATOMIC_RELAXED) == 1);

  update_pq_top_key(pq);
  __atomic_store_n(&_pq_locks[pq], 0, __ATOMIC_RELEASE);
}

void MultiQueueOverloadBalancer::update_pq_top_key(const std::size_t pq) {
  _pq_top_keys[pq] = _pqs[pq].empty() ? std::numeric_limits<float>::lowest() : _pqs[pq].peek_key();
}

void MultiQueueOverloadBalancer::clear_pqs() {
  for (std::size_t pq = 0; pq < _pqs.size(); ++pq) {
    lock_pq(pq);
    _pqs[pq].clear();
    unlock_pq(pq);
  }
}

} // namespace kaminpar::shm
