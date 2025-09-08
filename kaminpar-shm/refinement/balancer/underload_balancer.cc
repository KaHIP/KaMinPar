/*******************************************************************************
 * MultiQueue-based balancer for greedy minimum block weight balancing.
 *
 * @file:   underload_balancer.cc
 * @author: Daniel Seemaier
 * @date:   11.06.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/balancer/underload_balancer.h"

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/relative_gain.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

} // namespace

UnderloadBalancer::UnderloadBalancer(const Context &ctx) : _ctx(ctx) {}

UnderloadBalancer::~UnderloadBalancer() = default;

std::string UnderloadBalancer::name() const {
  return "Underload Balancer";
}

void UnderloadBalancer::initialize(const PartitionedGraph &) {
  // Nothing to do
}

bool UnderloadBalancer::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_TIMER("Underload Balancer");
  SCOPED_HEAP_PROFILER("Underload Balancer");

  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  // Terminate immediately if there is nothing to do
  if (!_p_ctx->has_min_block_weights() || metrics::is_min_balanced(*_p_graph, *_p_ctx)) {
    DBG << "Nothing to do: minimum block weights already satisfied.";
    return false;
  }

  _is_underloaded.resize(p_graph.k());
  _mq.reset(_ctx.parallel.num_threads, /* seed = */ 42);
  _node_target.resize(p_graph.n());
  _block_locks.clear();
  _block_locks.resize(p_graph.k());

  init_underloaded_blocks();

  reified(*_p_graph, [&]<typename Graph>(const Graph &graph) {
    _gain_cache.emplace<Graph>(_ctx, p_graph.k(), p_graph.k()).initialize(graph, p_graph);

    init_pqs(graph);
  });

  // Rebalance partition in parallel
  tbb::task_group tg;
  for (int task_id = 0; task_id < _ctx.parallel.num_threads; ++task_id) {
    tg.run([&] {
      reified(*_p_graph, [&](const auto &graph) { rebalance_worker(graph, task_id); });
    });
  }
  tg.wait();

  IF_DBG {
    BlockID num_underloaded_blocks = 0;
    BlockID num_overloaded_blocks = 0;
    std::vector<std::pair<BlockID, BlockWeight>> underloads;
    std::vector<std::pair<BlockID, BlockWeight>> overloads;

    for (const BlockID b : _p_graph->blocks()) {
      if (_p_graph->block_weight(b) < _p_ctx->min_block_weight(b)) {
        ++num_underloaded_blocks;
        underloads.emplace_back(b, _p_ctx->min_block_weight(b) - _p_graph->block_weight(b));
      }
      if (_p_graph->block_weight(b) > _p_ctx->max_block_weight(b)) {
        ++num_overloaded_blocks;
        overloads.emplace_back(b, _p_graph->block_weight(b) - _p_ctx->max_block_weight(b));
      }
    }
    DBG << "Result: there are now " << num_underloaded_blocks << " underloaded blocks and "
        << num_overloaded_blocks << " overloaded blocks";
    DBG << "Underloaded blocks: " << (underloads.empty() ? "(none)" : "");
    for (const auto &[b, delta] : underloads) {
      DBG << "\t" << b << ": " << delta;
    }
    DBG << "Overloaded blocks: " << (overloads.empty() ? "(none)" : "");
    for (const auto &[b, delta] : overloads) {
      DBG << "\t" << b << ": " << delta;
    }
  }

  return true;
}

template <typename Graph>
void UnderloadBalancer::rebalance_worker(const Graph &graph, [[maybe_unused]] const int thread_id) {
  DBG << "Worker " << thread_id << " started work";

  auto &gain_cache = _gain_cache.get<Graph>();

  while (auto pq = _mq.lock_pop_pq()) {
    KASSERT(!pq->empty());

    const NodeID node = pq->peek_id();
    const float gain = pq->peek_key();
    pq->pop();
    _mq.unlock(std::move(pq));

    const BlockID from = _p_graph->block(node);
    const BlockID to = _node_target[node];
    KASSERT(to != from);

    const NodeWeight weight = _p_graph->node_weight(node);

    // If the source block of node cannot afford to lose the node's weight, do not touch the node
    // again
    if (!is_movable_from(from, weight)) {
      continue;
    }

    const auto [actual_to, actual_gain] = compute_best_gain(graph, gain_cache, node, from);
    if (actual_to == kInvalidBlockID) {
      continue;
    }

    if (is_movable_to(to, weight) && actual_to == to && actual_gain >= gain) {
      lock_block(from);

      // Check weight of source block once more after locking it
      // We use locking since we cannot enforce both min and max weights using CAS; move() will
      // enforce max weight using CAS
      if (!is_movable_from(from, weight)) {
        unlock_block(from);
        continue;
      }

      if (_p_graph->move(
              node, from, to, _p_ctx->max_block_weight(to), _p_ctx->min_block_weight(from)
          )) {
        unlock_block(from);

        // Move successful: check whether the target block is still underloaded
        if (_p_graph->block_weight(to) >= _p_ctx->min_block_weight(to)) {
          __atomic_store_n(&_is_underloaded[to], 0u, __ATOMIC_RELAXED);
        }
      } else {
        unlock_block(from);
        insert_node_into_pq(node, actual_to, actual_gain);
      }
    } else {
      insert_node_into_pq(node, actual_to, actual_gain);
    }
  }

  DBG << "Worker " << thread_id << " terminated";
}

void UnderloadBalancer::init_underloaded_blocks() {
  [[maybe_unused]] BlockID num_underloaded_blocks = 0;

  for (const BlockID b : _p_graph->blocks()) {
    _is_underloaded[b] = (_p_graph->block_weight(b) < _p_ctx->min_block_weight(b));
    IFDBG(num_underloaded_blocks += _is_underloaded[b]);
  }

  DBG << num_underloaded_blocks << " out of " << _p_graph->k() << " blocks are underloaded";
}

template <typename Graph> void UnderloadBalancer::init_pqs(const Graph &graph) {
  GainCache<Graph> &gain_cache = _gain_cache.get<Graph>();

  [[maybe_unused]] std::atomic<NodeID> num_initial_nodes = 0;
  [[maybe_unused]] std::atomic<NodeID> num_rejected_nodes = 0;

  graph.pfor_nodes([&](const NodeID node) {
    const BlockID from = _p_graph->block(node);
    const NodeWeight weight = graph.node_weight(node);

    if (is_movable_from(from, weight)) {
      const auto [to, gain] = compute_best_gain(graph, gain_cache, node, from);
      if (to != kInvalidBlockID) {
        insert_node_into_pq(node, to, gain);
        IFDBG(++num_initial_nodes);
      } else {
        IFDBG(++num_rejected_nodes);
      }
    }
  });

  DBG << "Initialized multi-queue with " << num_initial_nodes << " nodes while skipping "
      << num_rejected_nodes << " heavy nodes";
}

void UnderloadBalancer::insert_node_into_pq(const NodeID node, const BlockID to, const float gain) {
  KASSERT(to != kInvalidBlockID);

  auto pq = _mq.lock_push_pq();
  pq->push(node, gain);
  _node_target[node] = to;
  _mq.unlock(std::move(pq));
}

std::pair<BlockID, float> UnderloadBalancer::compute_best_gain(
    const auto &graph, auto &gain_cache, const NodeID node, const BlockID from
) {
  const NodeWeight weight = graph.node_weight(node);
  BlockID best_block = kInvalidBlockID;
  EdgeWeight best_gain = std::numeric_limits<EdgeWeight>::min();

  gain_cache.gains(node, from, [&](const BlockID to, auto &&gain_fn) {
    if (is_movable_to(to, weight)) {
      const EdgeWeight gain = gain_fn();
      if (gain >= best_gain) {
        best_block = to;
        best_gain = gain;
      }
    }
  });

  return std::make_pair(best_block, compute_relative_gain(best_gain, weight));
}

bool UnderloadBalancer::is_movable_to(const BlockID block, const NodeWeight node_weight) const {
  KASSERT(block < _p_graph->k());

  return _is_underloaded[block] &&
         _p_graph->block_weight(block) + node_weight <= _p_ctx->max_block_weight(block);
}

bool UnderloadBalancer::is_movable_from(const BlockID from, const NodeWeight node_weight) const {
  return !_is_underloaded[from] &&
         _p_graph->block_weight(from) - node_weight >= _p_ctx->min_block_weight(from);
}

void UnderloadBalancer::lock_block(const BlockID block) {
  KASSERT(static_cast<std::size_t>(block) < _block_locks.size());

  std::uint8_t zero = 0u;
  while (!__atomic_compare_exchange_n(
      &_block_locks[block], &zero, 1u, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
  )) {
    zero = 0u;
  }
}

void UnderloadBalancer::unlock_block(const BlockID block) {
  KASSERT(static_cast<std::size_t>(block) < _block_locks.size());
  KASSERT(__atomic_load_n(&_block_locks[block], __ATOMIC_RELAXED) == 1u);

  __atomic_store_n(&_block_locks[block], 0u, __ATOMIC_RELAXED);
}

} // namespace kaminpar::shm
