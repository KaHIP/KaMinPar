/*******************************************************************************
 * Gain cache that caches one gain for each node and block, using a total of
 * O(|V| * k) memory.
 *
 * @file:   sparse_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class SparseDeltaGainCache;

template <bool iterate_nonadjacent_blocks = true, bool iterate_exact_gains = false>
class SparseGainCache {
  SET_DEBUG(true);

  using Self = SparseGainCache<iterate_nonadjacent_blocks, iterate_exact_gains>;
  template <typename, typename> friend class SparseDeltaGainCache;

public:
  template <typename DeltaPartitionedGraph>
  using DeltaCache = SparseDeltaGainCache<DeltaPartitionedGraph, Self>;

  // gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;

  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block.
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  SparseGainCache(
      const Context & /* ctx */, const NodeID preallocate_n, const BlockID preallocate_k
  )
      : _gain_cache(static_array::noinit, 1ull * preallocate_n * preallocate_k),
        _weighted_degrees(static_array::noinit, preallocate_n) {
    DBG << "Pre-allocating sparse gain cache: " << preallocate_n << " nodes, " << preallocate_k
        << " blocks -> allocate " << preallocate_n * preallocate_k * sizeof(EdgeWeight) / 1024
        << " KiB";
  }

  void initialize(const PartitionedGraph &p_graph) {
    _n = p_graph.n();
    _k = p_graph.k();

    const std::size_t gc_size = 1ull * _n * _k;

    if (_gain_cache.size() < gc_size) {
      SCOPED_TIMER("Allocation");
      DBG << "Re-allocating sparse gain cache: " << _n << " nodes, " << _k << " blocks -> allocate "
          << gc_size * sizeof(EdgeWeight) / 1024 << " KiB";
      _gain_cache.resize(static_array::noinit, gc_size);
    }
    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(static_array::noinit, _n);
    }

    reset();
    recompute_all(p_graph);
  }

  void free() {
    tbb::parallel_invoke([&] { _gain_cache.free(); }, [&] { _weighted_degrees.free(); });
  }

  [[nodiscard]] EdgeWeight
  gain(const NodeID node, const BlockID block_from, const BlockID block_to) const {
    return weighted_degree_to(node, block_to) - weighted_degree_to(node, block_from);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
  }

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    return weighted_degree_to(node, block);
  }

  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    const EdgeWeight conn_from = kIteratesExactGains ? conn(node, from) : 0;

    if constexpr (kIteratesNonadjacentBlocks) {
      for (BlockID to = 0; to < _k; ++to) {
        if (from != to) {
          lambda(to, [&] { return conn(node, to) - conn_from; });
        }
      }
    } else {
      for (BlockID to = 0; to < _k; ++to) {
        if (from != to) {
          const EdgeWeight conn_to = conn(node, to);
          if (conn_to > 0) {
            lambda(to, [&] { return conn_to - conn_from; });
          }
        }
      }
    }
  }

  void move(
      const PartitionedGraph &p_graph,
      const NodeID node,
      const BlockID block_from,
      const BlockID block_to
  ) {
    for (const auto &[e, v] : p_graph.neighbors(node)) {
      const EdgeWeight weight = p_graph.edge_weight(e);
      __atomic_fetch_sub(&_gain_cache[index(v, block_from)], weight, __ATOMIC_RELAXED);
      __atomic_fetch_add(&_gain_cache[index(v, block_to)], weight, __ATOMIC_RELAXED);
    }
  }

  [[nodiscard]] bool is_border_node(const NodeID node, const BlockID block) const {
    KASSERT(node < _weighted_degrees.size());
    return _weighted_degrees[node] != weighted_degree_to(node, block);
  }

  [[nodiscard]] bool validate(const PartitionedGraph &p_graph) const {
    bool valid = true;
    p_graph.pfor_nodes([&](const NodeID u) {
      if (!check_cached_gain_for_node(p_graph, u)) {
        LOG_WARNING << "gain cache invalid for node " << u;
        valid = false;
      }
    });
    return valid;
  }

  void print_statistics() const {
    // print statistics
  }

private:
  [[nodiscard]] EdgeWeight weighted_degree_to(const NodeID node, const BlockID block) const {
    KASSERT(index(node, block) < _gain_cache.size());
    return __atomic_load_n(&_gain_cache[index(node, block)], __ATOMIC_RELAXED);
  }

  [[nodiscard]] std::size_t index(const NodeID node, const BlockID block) const {
    const std::size_t idx = 1ull * node * _k + block;
    KASSERT(idx < _gain_cache.size());
    return idx;
  }

  void reset() {
    SCOPED_TIMER("Reset gain cache");
    tbb::parallel_for<std::size_t>(0, 1ull * _n * _k, [this](const std::size_t i) {
      _gain_cache[i] = 0;
    });
  }

  void recompute_all(const PartitionedGraph &p_graph) {
    SCOPED_TIMER("Recompute gain cache");
    p_graph.pfor_nodes([&](const NodeID u) { recompute_node(p_graph, u); });
  }

  void recompute_node(const PartitionedGraph &p_graph, const NodeID u) {
    KASSERT(u < p_graph.n());
    KASSERT(p_graph.block(u) < p_graph.k());

    const BlockID block_u = p_graph.block(u);
    _weighted_degrees[u] = 0;

    for (const auto &[e, v] : p_graph.neighbors(u)) {
      const BlockID block_v = p_graph.block(v);
      const EdgeWeight weight = p_graph.edge_weight(e);

      _gain_cache[index(u, block_v)] += weight;
      _weighted_degrees[u] += weight;
    }
  }

  [[nodiscard]] bool
  check_cached_gain_for_node(const PartitionedGraph &p_graph, const NodeID u) const {
    const BlockID block_u = p_graph.block(u);
    std::vector<EdgeWeight> actual_external_degrees(_k, 0);
    EdgeWeight actual_weighted_degree = 0;

    for (const auto &[e, v] : p_graph.neighbors(u)) {
      const BlockID block_v = p_graph.block(v);
      const EdgeWeight weight = p_graph.edge_weight(e);

      actual_weighted_degree += weight;
      actual_external_degrees[block_v] += weight;
    }

    for (BlockID b = 0; b < _k; ++b) {
      if (actual_external_degrees[b] != weighted_degree_to(u, b)) {
        LOG_WARNING << "For node " << u << ": cached weighted degree to block " << b << " is "
                    << weighted_degree_to(u, b) << " but should be " << actual_external_degrees[b];
        return false;
      }
    }

    if (actual_weighted_degree != _weighted_degrees[u]) {
      LOG_WARNING << "For node " << u << ": cached weighted degree is " << _weighted_degrees[u]
                  << " but should be " << actual_weighted_degree;
      return false;
    }

    return true;
  }

  NodeID _n;
  BlockID _k;

  StaticArray<EdgeWeight> _gain_cache;
  StaticArray<EdgeWeight> _weighted_degrees;
};

template <typename _DeltaPartitionedGraph, typename _GainCache> class SparseDeltaGainCache {
public:
  using DeltaPartitionedGraph = _DeltaPartitionedGraph;
  using GainCache = _GainCache;

  // Delta gain caches can only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph.
  static_assert(GainCache::kIteratesNonadjacentBlocks);
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  SparseDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph & /* d_graph */)
      : _gain_cache(gain_cache) {}

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn(node, block) + conn_delta(node, block);
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain(node, from, to) + conn_delta(node, to) - conn_delta(node, from);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    const EdgeWeight conn_from_delta = kIteratesExactGains ? conn_delta(node, from) : 0;

    _gain_cache.gains(node, from, [&](const BlockID to, auto &&gain) {
      lambda(to, [&] { return gain() + conn_delta(node, to) - conn_from_delta; });
    });
  }

  void move(
      const DeltaPartitionedGraph &d_graph,
      const NodeID u,
      const BlockID block_from,
      const BlockID block_to
  ) {
    for (const auto &[e, v] : d_graph.neighbors(u)) {
      const EdgeWeight weight = d_graph.edge_weight(e);
      _gain_cache_delta[_gain_cache.index(v, block_from)] -= weight;
      _gain_cache_delta[_gain_cache.index(v, block_to)] += weight;
    }
  }

  void clear() {
    _gain_cache_delta.clear();
  }

private:
  [[nodiscard]] EdgeWeight conn_delta(const NodeID node, const BlockID block) const {
    const auto it = _gain_cache_delta.get_if_contained(_gain_cache.index(node, block));
    return it != _gain_cache_delta.end() ? *it : 0;
  }

  const GainCache &_gain_cache;
  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;
};
} // namespace kaminpar::shm
