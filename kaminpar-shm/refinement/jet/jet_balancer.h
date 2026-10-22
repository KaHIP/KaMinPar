/*******************************************************************************
 * Bucket-based weak and strong rebalancing from Gilbert et al.,
 * "Jet: Multilevel Graph Partitioning on Graphics Processing Units", Section 4.2.
 *
 * @file:   jet_balancer.h
 * @author: Daniel Seemaier
 * @date:   22.10.2026
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace jet {

// Equation (5): negative losses first, then zero, then logarithmic positive-loss buckets.
inline std::size_t loss_bucket(const EdgeWeight loss) {
  if (loss < 0) {
    return 0;
  }
  if (loss == 0) {
    return 1;
  }
  return 1 + std::bit_width(static_cast<std::make_unsigned_t<EdgeWeight>>(loss));
}

} // namespace jet

template <typename Graph, typename GainCache> class JetBalancer {
  static constexpr std::size_t kNumBuckets = std::numeric_limits<EdgeWeight>::digits + 2;
  static constexpr std::size_t kInvalidBucket = std::numeric_limits<std::size_t>::max();

public:
  struct Statistics {
    std::size_t weak_rounds = 0;
    std::size_t strong_rounds = 0;
    std::size_t fallback_moves = 0;
  };

  explicit JetBalancer(const Context &) {}

  void initialize(const PartitionedGraph &) {}

  [[nodiscard]] const Statistics &statistics() const {
    return _statistics;
  }

  // Returns whether any vertex was moved. As with other balancers, weighted packing may be
  // infeasible: callers must check balance before accepting the partition.
  bool refine(
      PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      GainCache &gain_cache
  ) {
    SCOPED_TIMER("Jet Balancer");
    BlockWeight overload = metrics::total_overload(p_graph, p_ctx);
    if (overload == 0 || p_graph.k() < 2) {
      return false;
    }

    _destinations.resize(graph.n());
    _buckets.resize(graph.n());
    _ranks.resize(graph.n());
    _selected.resize(graph.n());
    bool changed = false;

    // The paper combines two weak passes with strong passes until balance is restored.
    for (int round = 0; round < 2 && overload > 0; ++round) {
      SCOPED_TIMER("Weak");
      ++_statistics.weak_rounds;
      changed |= perform_round(p_graph, graph, p_ctx, gain_cache, false);
      overload = metrics::total_overload(p_graph, p_ctx);
    }

    while (overload > 0) {
      SCOPED_TIMER("Strong");
      ++_statistics.strong_rounds;
      changed |= perform_round(p_graph, graph, p_ctx, gain_cache, true);
      const BlockWeight next_overload = metrics::total_overload(p_graph, p_ctx);
      if (next_overload >= overload) {
        break;
      }
      overload = next_overload;
    }

    // Indivisible coarse vertices may not fit in the deadzone or may be excluded by the weight
    // restriction. Finish such cases with capacity-safe moves, without the deadzone restriction.
    if (metrics::total_overload(p_graph, p_ctx) > 0) {
      SCOPED_TIMER("Weighted fallback");
      changed |= finish_weighted(p_graph, graph, p_ctx, gain_cache);
    }
    return changed;
  }

private:
  bool perform_round(
      PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      GainCache &gain_cache,
      const bool strong
  ) {
    const BlockID k = p_graph.k();
    _weights.resize(k);
    _limits.resize(k);
    _overloaded_index.assign(k, kInvalidBlockID);
    _overloaded.clear();
    _targets.clear();
    for (const BlockID b : p_graph.blocks()) {
      const BlockWeight weight = p_graph.block_weight(b);
      const BlockWeight limit = p_ctx.max_block_weight(b);
      const BlockWeight perfect = std::min(limit, p_ctx.perfectly_balanced_block_weight(b));
      _weights[b] = weight;

      // The paper leaves maxDestSize unspecified. These are the reference implementation's
      // weak/strong choices, clamped to the actual bound for small and nonuniform blocks.
      const BlockWeight fraction = static_cast<BlockWeight>(0.99L * limit);
      _limits[b] = std::min(
          limit,
          strong ? std::max(perfect + (perfect < limit), fraction)
                 : std::max(fraction, limit - std::min<BlockWeight>(limit, 100))
      );
      if (weight > limit) {
        _overloaded_index[b] = static_cast<BlockID>(_overloaded.size());
        _overloaded.push_back(b);
      } else if (weight < _limits[b]) {
        _targets.push_back(b);
      }
    }
    if (_overloaded.empty() || _targets.empty()) {
      return false;
    }

    // Only overloaded blocks need buckets; minibuckets reduce contention without O(k*n) storage.
    const std::size_t sections =
        std::min<std::size_t>(8, std::max<std::size_t>(1, graph.n() / (4096 * _overloaded.size())));
    const std::size_t width = kNumBuckets * sections;
    _bucket_offsets.assign(_overloaded.size() * width, 0);
    _evicted_weights.assign(_overloaded.size(), 0);

    graph.pfor_nodes([&](const NodeID u) {
      _buckets[u] = kInvalidBucket;
      _selected[u] = false;
      const BlockID from = p_graph.block(u);
      const BlockID index = _overloaded_index[from];
      const NodeWeight weight = graph.node_weight(u);
      if (index == kInvalidBlockID ||
          static_cast<long double>(weight) >
              1.5L * (_weights[from] - p_ctx.perfectly_balanced_block_weight(from)) ||
          weight > _weights[from] - p_ctx.min_block_weight(from)) {
        return;
      }

      const EdgeWeight conn_from = gain_cache.conn(u, from);
      EdgeWeight best_conn = 0;
      BlockID best = kInvalidBlockID;
      UnsignedEdgeWeight total_conn = 0;
      std::size_t num_adjacent_targets = 0;
      gain_cache.gains(u, from, [&](const BlockID to, auto &&value) {
        if (to == from || _weights[to] >= _limits[to]) {
          return;
        }
        const EdgeWeight conn = value() + (GainCache::kIteratesExactGains ? conn_from : 0);
        if (conn <= 0) {
          return;
        }
        total_conn += conn;
        ++num_adjacent_targets;
        if (conn > best_conn || (conn == best_conn && to < best)) {
          best = to;
          best_conn = conn;
        }
      });
      if (best == kInvalidBlockID) {
        best = strong ? _targets.front()
                      : _targets[Random::instance().random_index(0, _targets.size())];
      }
      _destinations[u] = best;
      const EdgeWeight conn_to = strong && num_adjacent_targets > 0
                                     ? static_cast<EdgeWeight>(total_conn / num_adjacent_targets)
                                     : best_conn;
      const std::size_t bucket =
          index * width + jet::loss_bucket(conn_from - conn_to) * sections + u % sections;
      _buckets[u] = bucket;
      _ranks[u] = __atomic_fetch_add(&_bucket_offsets[bucket], weight, __ATOMIC_RELAXED);
    });

    // Exclusive scans within each source block turn the per-bucket ranks into loss-ordered ranks.
    tbb::parallel_for<std::size_t>(0, _overloaded.size(), [&](const std::size_t index) {
      BlockWeight prefix = 0;
      for (std::size_t b = index * width; b < (index + 1) * width; ++b) {
        const BlockWeight weight = _bucket_offsets[b];
        _bucket_offsets[b] = prefix;
        prefix += weight;
      }
    });
    graph.pfor_nodes([&](const NodeID u) {
      if (_buckets[u] == kInvalidBucket) {
        return;
      }
      const BlockID from = p_graph.block(u);
      const BlockWeight rank = _ranks[u] + _bucket_offsets[_buckets[u]];
      _ranks[u] = rank;
      // Include the vertex crossing the overload threshold (exclusive prefix), as in the
      // reference implementation. A lower bound can prevent this last indivisible eviction.
      if (rank < _weights[from] - p_ctx.max_block_weight(from) &&
          rank + graph.node_weight(u) <= _weights[from] - p_ctx.min_block_weight(from)) {
        _selected[u] = true;
        __atomic_fetch_add(
            &_evicted_weights[_overloaded_index[from]], graph.node_weight(u), __ATOMIC_RELAXED
        );
      }
    });

    if (strong) {
      // Overlay contiguous destination-capacity intervals on the concatenated eviction lists.
      // Boundary-straddling weighted vertices are deferred rather than overfilling a destination.
      _eviction_offsets.resize(_overloaded.size());
      BlockWeight prefix = 0;
      for (std::size_t i = 0; i < _overloaded.size(); ++i) {
        _eviction_offsets[i] = prefix;
        prefix += _evicted_weights[i];
      }
      _capacity_offsets.resize(_targets.size() + 1);
      _capacity_offsets[0] = 0;
      for (std::size_t i = 0; i < _targets.size(); ++i) {
        const BlockID to = _targets[i];
        _capacity_offsets[i + 1] = _capacity_offsets[i] + _limits[to] - _weights[to];
      }
      graph.pfor_nodes([&](const NodeID u) {
        if (!_selected[u]) {
          return;
        }
        const BlockWeight rank = _ranks[u] + _eviction_offsets[_overloaded_index[p_graph.block(u)]];
        const auto end = std::upper_bound(_capacity_offsets.begin(), _capacity_offsets.end(), rank);
        if (end == _capacity_offsets.end() || graph.node_weight(u) > *end - rank) {
          _selected[u] = false;
        } else {
          _destinations[u] = _targets[end - _capacity_offsets.begin() - 1];
        }
      });
    }

    std::size_t num_moves = 0;
    graph.pfor_nodes([&](const NodeID u) {
      if (_selected[u]) {
        const BlockID from = p_graph.block(u);
        const BlockID to = _destinations[u];
        p_graph.set_block(u, to);
        gain_cache.move(u, from, to);
        __atomic_fetch_add(&num_moves, 1, __ATOMIC_RELAXED);
      }
    });
    return num_moves > 0;
  }

  bool finish_weighted(
      PartitionedGraph &p_graph,
      const Graph &graph,
      const PartitionContext &p_ctx,
      GainCache &gain_cache
  ) {
    // A fitting-capacity index avoids scanning all k blocks for each disconnected vertex.
    using Capacities = std::multimap<BlockWeight, BlockID>;
    Capacities capacities;
    std::vector<typename Capacities::iterator> positions(p_graph.k(), capacities.end());
    const auto update_capacity = [&](const BlockID b) {
      if (positions[b] != capacities.end()) {
        capacities.erase(positions[b]);
      }
      const BlockWeight capacity = p_ctx.max_block_weight(b) - p_graph.block_weight(b);
      positions[b] = capacity > 0 ? capacities.emplace(capacity, b) : capacities.end();
    };
    for (const BlockID b : p_graph.blocks()) {
      update_capacity(b);
    }

    bool changed = false;
    for (const NodeID u : graph.nodes()) {
      const BlockID from = p_graph.block(u);
      const NodeWeight weight = graph.node_weight(u);
      if (p_graph.block_weight(from) <= p_ctx.max_block_weight(from) ||
          weight > p_graph.block_weight(from) - p_ctx.min_block_weight(from)) {
        continue;
      }
      const auto fitting = capacities.lower_bound(weight);
      if (fitting == capacities.end()) {
        continue;
      }
      BlockID best = fitting->second;
      EdgeWeight best_conn = 0;
      const EdgeWeight conn_from = gain_cache.conn(u, from);
      gain_cache.gains(u, from, [&](const BlockID to, auto &&value) {
        const EdgeWeight conn = value() + (GainCache::kIteratesExactGains ? conn_from : 0);
        if (to != from && p_graph.block_weight(to) + weight <= p_ctx.max_block_weight(to) &&
            (conn > best_conn || (conn == best_conn && to < best))) {
          best = to;
          best_conn = conn;
        }
      });
      if (p_graph.move(u, from, best, p_ctx.max_block_weight(best), p_ctx.min_block_weight(from))) {
        gain_cache.move(u, from, best);
        update_capacity(from);
        update_capacity(best);
        ++_statistics.fallback_moves;
        changed = true;
      }
    }
    return changed;
  }

  Statistics _statistics;
  std::vector<BlockID> _destinations;
  std::vector<std::size_t> _buckets;
  std::vector<BlockWeight> _ranks;
  std::vector<std::uint8_t> _selected;
  std::vector<BlockWeight> _weights;
  std::vector<BlockWeight> _limits;
  std::vector<BlockID> _overloaded_index;
  std::vector<BlockID> _overloaded;
  std::vector<BlockID> _targets;
  std::vector<BlockWeight> _bucket_offsets;
  std::vector<BlockWeight> _evicted_weights;
  std::vector<BlockWeight> _eviction_offsets;
  std::vector<BlockWeight> _capacity_offsets;
};

} // namespace kaminpar::shm
