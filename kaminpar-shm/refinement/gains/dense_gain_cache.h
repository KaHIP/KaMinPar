/*******************************************************************************
 * Gain cache that caches one gain for each node and block, using a total of
 * O(|V| * k) memory.
 *
 * @file:   dense_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <typename> class DenseDeltaGainCache;

template <
    typename GraphType,
    bool iterate_nonadjacent_blocks = true,
    bool iterate_exact_gains = false>
class DenseGainCache {
  SET_DEBUG(true);

  template <typename> friend class DenseDeltaGainCache;

public:
  using Graph = GraphType;
  using Self = DenseGainCache<Graph, iterate_nonadjacent_blocks, iterate_exact_gains>;
  using DeltaGainCache = DenseDeltaGainCache<Self>;

  // gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;

  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block.
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  DenseGainCache(const Context & /* ctx */, const NodeID preallocate_n, const BlockID preallocate_k)
      : _gain_cache(1ull * preallocate_n * preallocate_k, static_array::noinit),
        _weighted_degrees(preallocate_n, static_array::noinit) {
    DBG << "Allocating gain cache: " << _gain_cache.size() * sizeof(EdgeWeight) << " bytes";
  }

  void initialize(const Graph &graph, const PartitionedGraph &p_graph) {
    _graph = &graph;
    _p_graph = &p_graph;

    _n = p_graph.n();
    _k = p_graph.k();

    const std::size_t gc_size = 1ull * _n * _k;
    if (_gain_cache.size() < gc_size) {
      SCOPED_TIMER("Allocation");
      _gain_cache.resize(gc_size, static_array::noinit);
      DBG << "Allocating gain cache: " << _gain_cache.size() * sizeof(EdgeWeight) << " bytes";
    }

    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(_n, static_array::noinit);
    }

    reset();
    recompute_all();
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

  void move(const NodeID node, const BlockID block_from, const BlockID block_to) {
    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      __atomic_fetch_sub(&_gain_cache[index(v, block_from)], weight, __ATOMIC_RELAXED);
      __atomic_fetch_add(&_gain_cache[index(v, block_to)], weight, __ATOMIC_RELAXED);
    });
  }

  [[nodiscard]] bool is_border_node(const NodeID node, const BlockID block) const {
    KASSERT(node < _weighted_degrees.size());
    return _weighted_degrees[node] != weighted_degree_to(node, block);
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

  void recompute_all() {
    SCOPED_TIMER("Recompute gain cache");
    _graph->pfor_nodes([&](const NodeID u) { recompute_node(u); });
  }

  void recompute_node(const NodeID u) {
    KASSERT(u < _n);
    KASSERT(_p_graph->block(u) < _k);

    const BlockID block_u = _p_graph->block(u);
    _weighted_degrees[u] = 0;

    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      const BlockID block_v = _p_graph->block(v);

      _gain_cache[index(u, block_v)] += weight;
      _weighted_degrees[u] += weight;
    });
  }

  const Graph *_graph = nullptr;
  const PartitionedGraph *_p_graph = nullptr;

  NodeID _n;
  BlockID _k;

  StaticArray<EdgeWeight> _gain_cache;
  StaticArray<EdgeWeight> _weighted_degrees;
};

template <typename GainCacheType> class DenseDeltaGainCache {
public:
  using GainCache = GainCacheType;

  // Delta gain caches can only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph.
  static_assert(GainCache::kIteratesNonadjacentBlocks);
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  DenseDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(d_graph) {}

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

  void move(const NodeID u, const BlockID block_from, const BlockID block_to) {
    _d_graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      _gain_cache_delta[_gain_cache.index(v, block_from)] -= weight;
      _gain_cache_delta[_gain_cache.index(v, block_to)] += weight;
    });
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
  const DeltaPartitionedGraph &_d_graph;
  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;
};

template <typename GraphType> using NormalDenseGainCache = DenseGainCache<GraphType>;

} // namespace kaminpar::shm
