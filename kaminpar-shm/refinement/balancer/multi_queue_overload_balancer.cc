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
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/relative_gain.h"
#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/gains/hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

template <typename GainCache> struct IsOnTheFlyGainCache : std::false_type {};

template <
    typename Graph,
    bool iterate_nonadjacent_blocks,
    bool iterate_exact_gains,
    bool iterate_source_block>
struct IsOnTheFlyGainCache<
    OnTheFlyGainCache<Graph, iterate_nonadjacent_blocks, iterate_exact_gains, iterate_source_block>>
    : std::true_type {};

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

  if (!begin_refinement(p_graph, p_ctx)) {
    return false;
  }

  reified(*_p_graph, [&]<typename Graph>(const Graph &graph) {
    auto &gain_cache = _gain_cache.emplace<Graph>(_ctx, p_graph.k(), p_graph.k());
    gain_cache.initialize(graph, p_graph);
    run_refinement(graph, gain_cache);
  });

  finish_refinement();

  return true;
}

bool MultiQueueOverloadBalancer::begin_refinement(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
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

  return true;
}

void MultiQueueOverloadBalancer::finish_refinement() {
  clear_pqs();
}

template <typename Graph, typename GainCacheT>
void MultiQueueOverloadBalancer::init_pqs(const Graph &graph, GainCacheT &gain_cache) {
  [[maybe_unused]] std::atomic<NodeID> num_initial_nodes = 0;
  [[maybe_unused]] std::atomic<NodeID> num_rejected_nodes = 0;

  using PQItems = std::vector<std::vector<std::pair<NodeID, float>>>;
  tbb::enumerable_thread_specific<PQItems> local_pq_items([&] { return PQItems(_pqs.size()); });

  std::atomic<int> seed{555};
  tbb::enumerable_thread_specific<AccessToken> tokens([&] {
    return AccessToken(seed.fetch_add(1, std::memory_order_relaxed), _pqs.size());
  });

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

    AccessToken &token = tokens.local();
    const std::size_t pq = token.pick_random_pq();

    _node_target[node] = to;
    __atomic_store_n(&_node_state[node], MOVABLE, __ATOMIC_RELAXED);
    _node_pq[node] = pq;
    local_pq_items.local()[pq].emplace_back(node, gain);
  });

  std::vector<PQItems *> local_pq_item_ptrs;
  for (PQItems &items : local_pq_items) {
    local_pq_item_ptrs.push_back(&items);
  }

  tbb::parallel_for<std::size_t>(0, _pqs.size(), [&](const std::size_t pq) {
    for (PQItems *items : local_pq_item_ptrs) {
      for (const auto &[node, gain] : (*items)[pq]) {
        _pqs[pq].push(node, gain);
        IFDBG(++num_initial_nodes);
      }
    }
    update_pq_top_key(pq);
  });

  DBG << "Initialized multi-queue overload balancer with " << num_initial_nodes
      << " candidate nodes while skipping " << num_rejected_nodes << " nodes";
}

template <typename Graph, typename GainCacheT>
void MultiQueueOverloadBalancer::rebalance_worker(
    const Graph &graph, GainCacheT &gain_cache, const int task_id
) {
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
      if constexpr (!IsOnTheFlyGainCache<std::remove_cvref_t<GainCacheT>>::value) {
        update_neighbors(graph, gain_cache, move);
      }
      if (_p_graph->block_weight(move.from) <= _p_ctx->max_block_weight(move.from)) {
        deactivate_overloaded_block(move.from);
      }
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
    const std::size_t pq = pq_top_key(first) < pq_top_key(second) ? second : first;

    if (!try_lock_pq(pq)) {
      continue;
    }

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
      const float key = pq_top_key(pq);
      if (key > best_key) {
        best_key = key;
        best_pq = pq;
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

    if (!try_lock_pq(best_pq)) {
      continue;
    }
    if (_pqs[best_pq].empty()) {
      unlock_pq(best_pq);
      continue;
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

    auto [neighbor_to, neighbor_gain] = compute_best_gain_of_candidates(
        graph, gain_cache, neighbor, neighbor_from, {_node_target[neighbor], move.from, move.to}
    );
    if (neighbor_to == kInvalidBlockID) {
      std::tie(neighbor_to, neighbor_gain) =
          compute_best_gain(graph, gain_cache, neighbor, neighbor_from);
    }
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

  auto consider_target = [&](const BlockID to, auto &&gain_fn) {
    if (to == from) {
      return;
    }
    const BlockWeight target_weight = _p_graph->block_weight(to);
    if (target_weight + weight <= _p_ctx->max_block_weight(to)) {
      const EdgeWeight gain = gain_fn();
      if (gain > best_gain || (gain == best_gain && target_weight < best_target_weight)) {
        best_block = to;
        best_gain = gain;
        best_target_weight = target_weight;
      }
    }
  };

  if constexpr (std::remove_reference_t<decltype(gain_cache)>::kIteratesNonadjacentBlocks) {
    gain_cache.gains(node, from, consider_target);
  } else {
    for (const BlockID to : _p_graph->blocks()) {
      consider_target(to, [&] { return gain_cache.gain(node, from, to); });
    }
  }

  if (best_block == kInvalidBlockID) {
    return {kInvalidBlockID, std::numeric_limits<float>::lowest()};
  }

  return {best_block, compute_relative_gain(best_gain, weight)};
}

std::pair<BlockID, float> MultiQueueOverloadBalancer::compute_best_gain_of_candidates(
    const auto &graph,
    const auto &gain_cache,
    const NodeID node,
    const BlockID from,
    const std::array<BlockID, 3> candidates
) {
  const NodeWeight weight = graph.node_weight(node);
  if (weight == 0) {
    return {kInvalidBlockID, std::numeric_limits<float>::lowest()};
  }

  BlockID best_block = kInvalidBlockID;
  EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();
  BlockWeight best_target_weight = _p_graph->block_weight(from) - weight;

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const BlockID to = candidates[i];
    if (to == kInvalidBlockID || to == from) {
      continue;
    }

    bool duplicate = false;
    for (std::size_t j = 0; j < i; ++j) {
      duplicate |= candidates[j] == to;
    }
    if (duplicate) {
      continue;
    }

    const BlockWeight target_weight = _p_graph->block_weight(to);
    if (target_weight + weight > _p_ctx->max_block_weight(to)) {
      continue;
    }

    const EdgeWeight gain = gain_cache.gain(node, from, to);
    if (gain > best_gain || (gain == best_gain && target_weight < best_target_weight)) {
      best_block = to;
      best_gain = gain;
      best_target_weight = target_weight;
    }
  }

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
  int spins = 0;
  while (!try_lock_pq(pq)) {
    if (++spins % 64 == 0) {
      std::this_thread::yield();
    }
  }
}

void MultiQueueOverloadBalancer::unlock_pq(const std::size_t pq) {
  KASSERT(pq < _pq_locks.size());
  KASSERT(__atomic_load_n(&_pq_locks[pq], __ATOMIC_RELAXED) == 1);

  update_pq_top_key(pq);
  __atomic_store_n(&_pq_locks[pq], 0, __ATOMIC_RELEASE);
}

float MultiQueueOverloadBalancer::pq_top_key(const std::size_t pq) const {
  KASSERT(pq < _pq_top_keys.size());
  return std::atomic_ref<float>(const_cast<float &>(_pq_top_keys[pq]))
      .load(std::memory_order_relaxed);
}

void MultiQueueOverloadBalancer::update_pq_top_key(const std::size_t pq) {
  const float top_key =
      _pqs[pq].empty() ? std::numeric_limits<float>::lowest() : _pqs[pq].peek_key();
  std::atomic_ref<float>(_pq_top_keys[pq]).store(top_key, std::memory_order_relaxed);
}

void MultiQueueOverloadBalancer::clear_pqs() {
  for (std::size_t pq = 0; pq < _pqs.size(); ++pq) {
    lock_pq(pq);
    _pqs[pq].clear();
    unlock_pq(pq);
  }
}

#define INSTANTIATE_CACHED_REBALANCER(Graph, GainCache)                                            \
  template void MultiQueueOverloadBalancer::init_pqs<Graph, GainCache>(                            \
      const Graph &, GainCache &                                                                   \
  );                                                                                               \
  template void MultiQueueOverloadBalancer::rebalance_worker<Graph, GainCache>(                    \
      const Graph &, GainCache &, int                                                              \
  )

using CSRNormalSparseGainCache = NormalSparseGainCache<const CSRGraph>;
using CSRNormalCompactHashingGainCache = NormalCompactHashingGainCache<const CSRGraph>;
using CSRLargeKCompactHashingGainCache = LargeKCompactHashingGainCache<const CSRGraph>;

using CompressedNormalSparseGainCache = NormalSparseGainCache<const CompressedGraph>;
using CompressedNormalCompactHashingGainCache =
    NormalCompactHashingGainCache<const CompressedGraph>;
using CompressedLargeKCompactHashingGainCache =
    LargeKCompactHashingGainCache<const CompressedGraph>;

INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRNormalSparseGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRNormalCompactHashingGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRLargeKCompactHashingGainCache);

INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedNormalSparseGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedNormalCompactHashingGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedLargeKCompactHashingGainCache);

#ifdef KAMINPAR_EXPERIMENTAL
using CSRLargeKSparseGainCache = LargeKSparseGainCache<const CSRGraph>;
using CSRNormalHashingGainCache = NormalHashingGainCache<const CSRGraph>;
using CSRLargeKHashingGainCache = LargeKHashingGainCache<const CSRGraph>;
using CSRNormalDenseGainCache = NormalDenseGainCache<const CSRGraph>;
using CSRLargeKDenseGainCache = LargeKDenseGainCache<const CSRGraph>;
using CSRNormalOnTheFlyGainCache = NormalOnTheFlyGainCache<const CSRGraph>;

using CompressedLargeKSparseGainCache = LargeKSparseGainCache<const CompressedGraph>;
using CompressedNormalHashingGainCache = NormalHashingGainCache<const CompressedGraph>;
using CompressedLargeKHashingGainCache = LargeKHashingGainCache<const CompressedGraph>;
using CompressedNormalDenseGainCache = NormalDenseGainCache<const CompressedGraph>;
using CompressedLargeKDenseGainCache = LargeKDenseGainCache<const CompressedGraph>;
using CompressedNormalOnTheFlyGainCache = NormalOnTheFlyGainCache<const CompressedGraph>;

INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRLargeKSparseGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRNormalHashingGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRLargeKHashingGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRNormalDenseGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRLargeKDenseGainCache);
INSTANTIATE_CACHED_REBALANCER(CSRGraph, CSRNormalOnTheFlyGainCache);

INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedLargeKSparseGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedNormalHashingGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedLargeKHashingGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedNormalDenseGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedLargeKDenseGainCache);
INSTANTIATE_CACHED_REBALANCER(CompressedGraph, CompressedNormalOnTheFlyGainCache);
#endif // KAMINPAR_EXPERIMENTAL

#undef INSTANTIATE_CACHED_REBALANCER

} // namespace kaminpar::shm
