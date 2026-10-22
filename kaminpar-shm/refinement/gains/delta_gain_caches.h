/*******************************************************************************
 * @file:   delta_gain_caches.h
 * @author: Daniel Seemaier
 * @date:   24.09.2024
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/flat_hash_map.h"
#include "kaminpar-common/inline.h"

namespace kaminpar::shm {

template <typename GainCacheType> class GenericDeltaGainCache {
public:
  using GainCache = GainCacheType;
  using Graph = typename GainCache::Graph;

  // Delta gain caches should only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph. These connections might be
  // missed if the gain cache does not iterate over all blocks.
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;
  static_assert(GainCache::kIteratesNonadjacentBlocks);

  GenericDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(d_graph),
        _k(d_graph.k()) {}

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn(node, block) + conn_delta(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain(node, from, to) + conn_delta(node, to) - conn_delta(node, from);
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
  }

  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    const EdgeWeight conn_from_delta = kIteratesExactGains ? conn_delta(node, from) : 0;

    _gain_cache.gains(node, from, [&](const BlockID to, auto &&gain) {
      lambda(to, [&] { return gain() + conn_delta(node, to) - conn_from_delta; });
    });
  }

  KAMINPAR_INLINE void move(const NodeID u, const BlockID block_from, const BlockID block_to) {
    reified(_d_graph, [&](const auto &graph) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        _gain_cache_delta[index(v, block_from)] -= weight;
        _gain_cache_delta[index(v, block_to)] += weight;
      });
    });
  }

  KAMINPAR_INLINE void clear() {
    _gain_cache_delta.clear();
  }

private:
  [[nodiscard]] KAMINPAR_INLINE std::size_t index(const NodeID node, const BlockID block) const {
    // Note: this increases running times substantially due to the shifts
    // return index_sparse(node, block);
    return 1ull * node * _k + block;
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_delta(const NodeID node, const BlockID block) const {
    const auto it = _gain_cache_delta.get_if_contained(index(node, block));
    return it != _gain_cache_delta.end() ? *it : 0;
  }

  const GainCache &_gain_cache;
  const DeltaPartitionedGraph &_d_graph;
  BlockID _k;

  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;
};

template <typename GainCacheType> class LargeKGenericDeltaGainCache {
public:
  using GainCache = GainCacheType;
  using Graph = typename GainCache::Graph;

  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  // If k is large, iterating over all blocks becomes very expensive -- this delta gain cache should
  // only be used when iterating over adjacent blocks only.
  static_assert(!GainCache::kIteratesNonadjacentBlocks);

  LargeKGenericDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(d_graph),
        _k(d_graph.k()) {}

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn(node, block) + conn_delta(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain(node, from, to) + conn_delta(node, to) - conn_delta(node, from);
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
  }

  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    const EdgeWeight conn_from_delta = kIteratesExactGains ? conn_delta(node, from) : 0;

    // Reuse sparse scratch storage and finish reading the shared row before invoking the
    // consumer. Some gain caches lock the row during gains(), so its callback must not
    // query the same cache again or keep the lock during the consumer's work.
    _gain_snapshot.clear();
    _gain_cache.gains(node, from, [&](const BlockID to, auto &&gain) {
      _gain_snapshot[to] = gain() + conn_delta(node, to) - conn_from_delta;
    });

    const auto it = _adjacent_blocks_delta.find(node);
    if (it != _adjacent_blocks_delta.end()) {
      for (const BlockID to : it->second) {
        if (to != from && !_gain_snapshot.contains(to)) {
          const EdgeWeight shared_value = [&] {
            if constexpr (kIteratesExactGains) {
              return _gain_cache.gain(node, from, to);
            } else {
              return _gain_cache.conn(node, to);
            }
          }();
          _gain_snapshot[to] = shared_value + conn_delta(node, to) - conn_from_delta;
        }
      }
    }

    _gain_snapshot.for_each([&](const BlockID to, const EdgeWeight value) {
      const EdgeWeight connectivity = kIteratesExactGains ? conn(node, to) : value;
      if (connectivity <= 0) {
        return;
      }
      lambda(to, [&] { return value; });
    });
  }

  KAMINPAR_INLINE void move(const NodeID u, const BlockID block_from, const BlockID block_to) {
    if (block_from == block_to) {
      return;
    }

    reified(_d_graph, [&](const auto &graph) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        add_delta(v, block_from, -weight);
        add_delta(v, block_to, weight);
      });
    });
  }

  KAMINPAR_INLINE void clear() {
    _gain_cache_delta.clear();
    _adjacent_blocks_delta.clear();
    _gain_snapshot.clear();
  }

private:
  KAMINPAR_INLINE void add_delta(const NodeID node, const BlockID block, const EdgeWeight delta) {
    const std::size_t key = index(node, block);
    // A shared connection can disappear after this local move. Track every touched block
    // independently of its current shared connectivity, including deltas that later cancel.
    if (!_gain_cache_delta.contains(key)) {
      _adjacent_blocks_delta[node].push_back(block);
    }
    _gain_cache_delta[key] += delta;
  }

  [[nodiscard]] KAMINPAR_INLINE std::size_t index(const NodeID node, const BlockID block) const {
    return 1ull * node * _k + block;
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_delta(const NodeID node, const BlockID block) const {
    const auto it = _gain_cache_delta.get_if_contained(index(node, block));
    return it != _gain_cache_delta.end() ? *it : 0;
  }

  const GainCache &_gain_cache;
  const DeltaPartitionedGraph &_d_graph;
  BlockID _k;

  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;

  FlatHashMap<NodeID, std::vector<BlockID>> _adjacent_blocks_delta;

  // Delta gain caches are worker-local. Queries are not reentrant; this map retains its
  // allocation across queries and iterates only entries used by the current query.
  mutable DynamicRememberingFlatMap<BlockID, EdgeWeight> _gain_snapshot;
};

} // namespace kaminpar::shm
