/*******************************************************************************
 * Gain cache wrapper that traces all operations and can dump them to disk.
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

namespace kaminpar::shm {
namespace tracing {
class Tracer {
  enum OperationType : std::uint8_t {
    GAIN_SINGLE,
    GAIN_DOUBLE,
    CONN,
    GAINS,
    MOVE,
    IS_BORDER,
  };

  struct Operation {
    const void *actor;
    OperationType type;
    std::array<std::uint64_t, 4> data;
  };

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

  void dump(const std::string &filename) {
    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::binary);
    out.write(reinterpret_cast<char *>(_trace.data()), sizeof(Operation) * _trace.size());
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
        actor,
        type,
        {a, b, c, d},
    });
  }

  std::vector<Operation> _trace;
  std::mutex _trace_mutex;
};
} // namespace tracing

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
  mutable tracing::Tracer _tracer;
  std::string _filename;
};

template <typename ActualDeltaGainCache, typename _TracingGainCache> class TracingDeltaGainCache {
public:
  using DeltaPartitionedGraph = typename ActualDeltaGainCache::DeltaPartitionedGraph;
  using GainCache = _TracingGainCache;

  constexpr static bool kIteratesNonadjacentBlocks =
      ActualDeltaGainCache::kIteratesNonadjacentBlocks;
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
