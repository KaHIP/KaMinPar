/*******************************************************************************
 * Gain cache wrapper that traces all operations and can dump them to disk /
 * load them from disk and replay them to some gain cache.
 *
 * @file:   tracing_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   27.02.2024
 ******************************************************************************/
#pragma once

#include <fstream>
#include <mutex>
#include <sstream>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
namespace tracing_gain_cache {
SET_DEBUG(true);

enum OperationType : std::uint8_t {
  GAIN_SINGLE,
  GAIN_DOUBLE,
  CONN,
  GAINS,
  MOVE,
  IS_BORDER,
  CLEAR,
  NUM_OPS,
};

struct Operation {
  void *actor;
  OperationType op;
  std::array<std::uint64_t, 4> data;
};

template <typename GainCache, typename DeltaPartitionedGraph>
std::vector<std::uint64_t>
replay(const std::string &filename, const Context &ctx, PartitionedGraph &p_graph) {
  using DeltaGainCache = typename GainCache::template DeltaCache<DeltaPartitionedGraph>;
  GainCache gain_cache(ctx, p_graph.n(), p_graph.k());
  gain_cache.initialize(p_graph);

  std::vector<Operation> queries;
  std::vector<std::uint64_t> ans;

  TIMED_SCOPE("Reading trace") {
    std::ifstream in(filename, std::ios_base::binary);

    std::uint64_t size;
    in.read(reinterpret_cast<char *>(&size), sizeof(std::uint64_t));

    queries.resize(size);
    ans.resize(size);

    in.read(reinterpret_cast<char *>(queries.data()), size * sizeof(Operation));
  };

  // Create delta gain caches
  using DeltaPair =
      std::pair<std::unique_ptr<DeltaPartitionedGraph>, std::unique_ptr<DeltaGainCache>>;
  std::unordered_map<void *, DeltaPair> delta_mapping;

  std::vector<std::size_t> counts(OperationType::NUM_OPS);

  START_TIMER("Preparing delta pairs");
  for (auto &query : queries) {
    ++counts[query.op];

    if (query.actor == nullptr) {
      continue;
    }

    if (delta_mapping.find(query.actor) == delta_mapping.end()) {
      auto d_graph = std::make_unique<DeltaPartitionedGraph>(&p_graph);
      auto d_gain_cache = std::make_unique<DeltaGainCache>(gain_cache, *d_graph);
      delta_mapping.emplace(
          query.actor, std::make_pair(std::move(d_graph), std::move(d_gain_cache))
      );
    }
    query.actor = reinterpret_cast<void *>(&delta_mapping[query.actor]);
  }
  STOP_TIMER();

  LOG << "Operations: " << queries.size();
  LOG << "  - GAIN_SINGLE: " << counts[OperationType::GAIN_SINGLE];
  LOG << "  - GAIN_DOUBLE: " << counts[OperationType::GAIN_DOUBLE];
  LOG << "  - CONN: " << counts[OperationType::CONN];
  LOG << "  - GAINS: " << counts[OperationType::GAINS];
  LOG << "  - MOVE: " << counts[OperationType::MOVE];
  LOG << "  - IS_BORDER: " << counts[OperationType::IS_BORDER];
  LOG << "  - CLEAR: " << counts[OperationType::CLEAR];
  LOG << "Delta gain caches: " << delta_mapping.size();

  START_TIMER("Replaying recorded trace");
  for (std::size_t i = 0; i < queries.size(); ++i) {
    const auto &query = queries[i];
    const OperationType op = query.op;
    const auto &[a, b, c, d] = query.data;

    if (query.actor == nullptr) {
      switch (query.op) {
      case GAIN_SINGLE:
        ans[i] = gain_cache.gain(a, b, c);
        break;

      case GAIN_DOUBLE: {
        const auto &[gain_c, gain_d] = gain_cache.gain(a, b, {c, d});
        ans[i] = gain_c + gain_d;
        break;
      }

      case CONN:
        ans[i] = gain_cache.conn(a, b);
        break;

      case GAINS:
        // do nothing
        break;

      case MOVE:
        gain_cache.move(p_graph, a, b, c);
        break;

      case IS_BORDER:
        ans[i] = gain_cache.is_border_node(a, b);
        break;

      case CLEAR:
        // do nothing
        break;

      case NUM_OPS:
        // do nothing
        break;
      }
    } else {
      const auto &[d_graph, d_gain_cache] = *reinterpret_cast<DeltaPair *>(query.actor);

      switch (query.op) {
      case GAIN_SINGLE:
        ans[i] = d_gain_cache->gain(a, b, c);
        break;

      case GAIN_DOUBLE: {
        const auto &[ans_c, ans_d] = d_gain_cache->gain(a, b, {c, d});
        ans[i] = ans_c + ans_d;
        break;
      }

      case CONN:
        ans[i] = d_gain_cache->conn(a, b);
        break;

      case GAINS:
        // do nothing
        break;

      case MOVE:
        d_gain_cache->move(*d_graph, a, b, c);
        break;

      case IS_BORDER:
        // do nothing
        break;

      case CLEAR:
        d_gain_cache->clear();
        break;

      case NUM_OPS:
        // do nothing
        break;
      }
    }
  }
  STOP_TIMER();

  return ans;
}

class Tracer {
public:
  void trace_gain(const void *cache, const NodeID node, const BlockID from, const BlockID to) {
    trace(cache, GAIN_SINGLE, node, from, to);
  }

  void trace_gain(
      const void *cache,
      const NodeID node,
      const BlockID from,
      const std::pair<BlockID, BlockID> &targets
  ) {
    trace(cache, GAIN_DOUBLE, node, from, targets.first, targets.second);
  }

  void trace_conn(const void *cache, const NodeID node, const BlockID to) {
    trace(cache, CONN, node, to);
  }

  void trace_gains(const void *cache, const NodeID node, const BlockID from) {
    trace(cache, GAINS, node, from);
  }

  void trace_move(const void *cache, const NodeID node, const BlockID from, const BlockID to) {
    trace(cache, MOVE, node, from, to);
  }

  void trace_is_border_node(const void *cache, const NodeID node, const BlockID block) {
    trace(cache, IS_BORDER, node, block);
  }

  void trace_clear(const void *cache) {
    trace(cache, CLEAR);
  }

  void dump(const std::string &filename) {
    DBG << "Dumping " << _trace.size() << " operations to " << filename;

    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::binary);
    std::uint64_t size = _trace.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(std::uint64_t));
    out.write(reinterpret_cast<const char *>(_trace.data()), sizeof(Operation) * _trace.size());
  }

  void clear() {
    _trace.clear();
  }

  [[nodiscard]] bool empty() {
    return _trace.empty();
  }

private:
  void trace(
      const void *actor,
      const OperationType type,
      const std::uint64_t a = 0,
      const std::uint64_t b = 0,
      const std::uint64_t c = 0,
      const std::uint64_t d = 0
  ) {
    std::scoped_lock<std::mutex> lk(_trace_mutex);
    _trace.push_back(Operation{
        const_cast<void *>(actor),
        type,
        {a, b, c, d},
    });
  }

  std::vector<Operation> _trace;
  std::mutex _trace_mutex;
};
} // namespace tracing_gain_cache

template <typename ActualDeltaGainCache, typename GainCache> class TracingDeltaGainCache;

template <typename ActualGainCache> class TracingGainCache {
  using Self = TracingGainCache<ActualGainCache>;
  template <typename, typename> friend class TracingDeltaGainCache;

public:
  template <typename DeltaPartitionedGraph>
  using ActualDeltaGainCache = typename ActualGainCache::template DeltaCache<DeltaPartitionedGraph>;

  template <typename DeltaPartitionedGraph>
  using DeltaCache = TracingDeltaGainCache<ActualDeltaGainCache<DeltaPartitionedGraph>, Self>;

  constexpr static bool kIteratesNonadjacentBlocks = ActualGainCache::kIteratesNonadjacentBlocks;
  constexpr static bool kIteratesExactGains = ActualGainCache::kIteratesExactGains;

  TracingGainCache(const Context &ctx, const NodeID preallocate_n, const BlockID preallocate_k)
      : _gain_cache(ctx, preallocate_n, preallocate_k) {}

  void initialize(const PartitionedGraph &p_graph) {
    if (!_tracer.empty()) {
      _tracer.dump(_filename);
      _tracer.clear();
    }

    _gain_cache.initialize(p_graph);

    std::stringstream ss;
    ss << "gaincache_n" << p_graph.n() << "_m" << p_graph.m() << "_k" << p_graph.k() << ".trace";
    _filename = ss.str();
  }

  void free() {
    if (!_tracer.empty()) {
      _tracer.dump(_filename);
      _tracer.clear();
    }
    _gain_cache.free();
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    _tracer.trace_gain(nullptr, node, from, to);
    return _gain_cache.gain(node, from, to);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    _tracer.trace_gain(nullptr, node, b_node, targets);
    return _gain_cache.gain(node, b_node, targets);
  }

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    _tracer.trace_conn(nullptr, node, block);
    return _gain_cache.conn(node, block);
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    _tracer.trace_gains(nullptr, node, from);
    _gain_cache.gains(node, from, std::forward<Lambda>(lambda));
  }

  void
  move(const PartitionedGraph &p_graph, const NodeID node, const BlockID from, const BlockID to) {
    _tracer.trace_move(nullptr, node, from, to);
    _gain_cache.move(p_graph, node, from, to);
  }

  [[nodiscard]] bool is_border_node(const NodeID node, const BlockID block) const {
    _tracer.trace_is_border_node(nullptr, node, block);
    return _gain_cache.is_border_node(node, block);
  }

  [[nodiscard]] bool validate(const PartitionedGraph &p_graph) const {
    return _gain_cache.validate(p_graph);
  }

  void print_statistics() const {
    _gain_cache.print_statistics();
  }

  [[nodiscard]] const ActualGainCache &get() const {
    return _gain_cache;
  }

  ActualGainCache _gain_cache;
  mutable tracing_gain_cache::Tracer _tracer;
  std::string _filename;
};

template <typename ActualDeltaGainCache, typename _TracingGainCache> class TracingDeltaGainCache {
public:
  using DeltaPartitionedGraph = typename ActualDeltaGainCache::DeltaPartitionedGraph;
  using GainCache = _TracingGainCache;

  constexpr static bool kIteratesExactGains = ActualDeltaGainCache::kIteratesExactGains;

  TracingDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_gain_cache(gain_cache.get(), d_graph) {}

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    _gain_cache._tracer.trace_conn(this, node, block);
    return _d_gain_cache.conn(node, block);
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    _gain_cache._tracer.trace_gain(this, node, from, to);
    return _d_gain_cache.gain(node, from, to);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    _gain_cache._tracer.trace_gain(this, node, b_node, targets);
    return _d_gain_cache.gain(node, b_node, targets);
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    _gain_cache._tracer.trace_gains(this, node, from);
    return _d_gain_cache.gains(node, from, std::forward<Lambda>(lambda));
  }

  void
  move(const DeltaPartitionedGraph &d_graph, const NodeID u, const BlockID from, const BlockID to) {
    _gain_cache._tracer.trace_move(this, u, from, to);
    _d_gain_cache.move(d_graph, u, from, to);
  }

  void clear() {
    _gain_cache._tracer.trace_clear(this);
    _d_gain_cache.clear();
  }

private:
  void *me() {
    return reinterpret_cast<void *>(this);
  }

  const GainCache &_gain_cache;
  ActualDeltaGainCache _d_gain_cache;
};
} // namespace kaminpar::shm
