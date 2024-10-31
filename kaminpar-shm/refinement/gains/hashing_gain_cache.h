/*******************************************************************************
 * @file:   hashing_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   13.08.2024
 ******************************************************************************/
#pragma once

#include <limits>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/delta_gain_caches.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/compact_hash_map.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

template <
    typename GraphType,
    template <typename> typename DeltaGainCacheType,
    bool iterate_nonadjacent_blocks,
    bool iterate_exact_gains = false>
class HashingGainCache {
  SET_DEBUG(false);

  // Abuse MSB bit in the _weighted_degrees[] array for locking
  constexpr static UnsignedEdgeWeight kWeightedDegreeLock =
      (static_cast<UnsignedEdgeWeight>(1) << (std::numeric_limits<UnsignedEdgeWeight>::digits - 1));
  constexpr static UnsignedEdgeWeight kWeightedDegreeMask = ~kWeightedDegreeLock;

public:
  using Graph = GraphType;

  using Self = HashingGainCache<
      GraphType,
      DeltaGainCacheType,
      iterate_nonadjacent_blocks,
      iterate_exact_gains>;

  using DeltaGainCache = DeltaGainCacheType<Self>;

  // gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;

  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block.
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  HashingGainCache(const Context &ctx, const NodeID preallocate_n, BlockID)
      : _ctx(ctx),
        // Since we do not know the size of the gain cache in advance (depends on vertex degrees),
        // we cannot preallocate it
        _gain_cache(0, static_array::noinit),
        _weighted_degrees(preallocate_n, static_array::noinit) {}

  void initialize(const Graph &graph, const PartitionedGraph &p_graph) {
    _graph = &graph;
    _p_graph = &p_graph;

    _n = _graph->n();
    _k = _p_graph->k();

    if (_offsets.size() < _n + 1) {
      SCOPED_TIMER("Allocation");
      _offsets.resize(_n + 1);
    }
    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(_n);
    }

    _offsets.front() = 0;
    _graph->pfor_nodes([&](const NodeID u) {
      _offsets[u + 1] = std::min<EdgeID>(math::ceil2(_graph->degree(u)), _k);
    });
    parallel::prefix_sum(_offsets.begin(), _offsets.begin() + _n + 1, _offsets.begin());
    const std::size_t gain_cache_size = _offsets.back();

    if (_gain_cache.size() < gain_cache_size) {
      SCOPED_TIMER("Allocation");
      _gain_cache.resize(gain_cache_size);
      DBG << "Allocating gain cache: " << _gain_cache.size() * sizeof(UnsignedEdgeWeight)
          << " bytes";
    }

    _bits_for_key = math::ceil_log2(_k);
    DBG << "Gain cache summary: have " << _n << " nodes, " << _k << " blocks";
    DBG << "  Reserve " << (sizeof(UnsignedEdgeWeight) * 8 - _bits_for_key) << " of "
        << sizeof(UnsignedEdgeWeight) * 8 << " bits for gain values";
    DBG << "  Reserve " << _bits_for_key << " of " << sizeof(UnsignedEdgeWeight) * 8
        << " bits for block IDs";

    reset();
    recompute();
  }

  void free() {
    _gain_cache.free();
    _weighted_degrees.free();
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain(const NodeID node, const BlockID block_from, const BlockID block_to) const {
    return weighted_degree_to(node, block_to) - weighted_degree_to(node, block_from);
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) {
    return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight conn(const NodeID node, const BlockID block) const {
    return weighted_degree_to(node, block);
  }

  // Forcing inlining here seems to be very important
  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    if (use_full_table(node)) {
      const EdgeWeight conn_from = kIteratesExactGains ? conn_full_table(node, from) : 0;

      for (BlockID to = 0; to < _k; ++to) {
        if (from == to) {
          continue;
        }

        if constexpr (kIteratesNonadjacentBlocks) {
          lambda(to, [&] { return conn_full_table(node, to) - conn_from; });
        } else {
          const EdgeWeight conn_to = conn_full_table(node, to);
          if (conn_to > 0) {
            lambda(to, [&] { return conn_to - conn_from; });
          }
        }
      }
    } else {
      const EdgeWeight conn_from = kIteratesExactGains ? conn_hash_table(node, from) : 0;

      if constexpr (kIteratesNonadjacentBlocks) {
        auto &buffer = _sparse_buffer_ets.local();

        create_hash_table(node).for_each([&](const BlockID to, const EdgeWeight conn_to) {
          buffer.set(to, conn_to);
        });

        for (BlockID to = 0; to < _k; ++to) {
          if (from != to) {
            lambda(to, [&] { return buffer.get(to) - conn_from; });
          }
        }

        buffer.clear();
      } else {
        create_hash_table(node).for_each([&](const BlockID to, const EdgeWeight conn_to) {
          if (to != from) {
            lambda(to, [&] { return conn_to - conn_from; });
          }
        });
      }
    }
  }

  KAMINPAR_INLINE void move(const NodeID node, const BlockID block_from, const BlockID block_to) {
    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      if (use_full_table(v)) {
        __atomic_fetch_sub(&_gain_cache[index_full_table(v, block_from)], weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_gain_cache[index_full_table(v, block_to)], weight, __ATOMIC_RELAXED);
      } else {
        lock(v);
        auto table = create_hash_table(v);
        table.decrease_by(block_from, weight);
        table.increase_by(block_to, weight);
        unlock(v);
      }
    });
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  is_border_node(const NodeID node, const BlockID block_of_node) const {
    return weighted_degree(node) != weighted_degree_to(node, block_of_node);
  }

  void print_statistics() const {}

private:
  //
  // Locking (hash table)
  //

  void lock(const NodeID node) {
    UnsignedEdgeWeight state = __atomic_load_n(&_weighted_degrees[node], __ATOMIC_RELAXED);
    do {
      while (state & kWeightedDegreeLock) {
        state = __atomic_load_n(&_weighted_degrees[node], __ATOMIC_RELAXED);
      }
    } while (!__atomic_compare_exchange_n(
        &_weighted_degrees[node],
        &state,
        state | kWeightedDegreeLock,
        false,
        __ATOMIC_ACQUIRE,
        __ATOMIC_RELAXED
    ));
  }

  void unlock(const NodeID node) {
    __atomic_store_n(
        &_weighted_degrees[node], _weighted_degrees[node] & kWeightedDegreeMask, __ATOMIC_RELAXED
    );
  }

  //
  // Lookups (mixed)
  //

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight weighted_degree(const NodeID node) const {
    KASSERT(node < _weighted_degrees.size());
    return static_cast<EdgeWeight>(_weighted_degrees[node] & kWeightedDegreeMask);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  weighted_degree_to(const NodeID node, const BlockID block) const {
    if (use_full_table(node)) {
      const std::size_t idx = index_full_table(node, block);
      return static_cast<EdgeWeight>(__atomic_load_n(&_gain_cache[idx], __ATOMIC_RELAXED));
    } else {
      return static_cast<EdgeWeight>(create_hash_table(node).get(block));
    }
  }

  //
  // Lookups (hash table)
  //

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_hash_table(const NodeID node, const BlockID block) const {
    return weighted_degree_to_hash_table(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  weighted_degree_to_hash_table(const NodeID node, const BlockID block) const {
    return static_cast<EdgeWeight>(create_hash_table(node).get(block));
  }

  [[nodiscard]] KAMINPAR_INLINE CompactHashMap<UnsignedEdgeWeight const, true>
  create_hash_table(const NodeID node) const {
    const std::size_t start = _offsets[node];
    const std::size_t size = _offsets[node + 1] - start;
    KASSERT(math::is_power_of_2(size));
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  [[nodiscard]] KAMINPAR_INLINE CompactHashMap<UnsignedEdgeWeight, true>
  create_hash_table(const NodeID node) {
    const std::size_t start = _offsets[node];
    const std::size_t size = _offsets[node + 1] - start;
    KASSERT(math::is_power_of_2(size));
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  //
  // Lookups (full table)
  //

  [[nodiscard]] KAMINPAR_INLINE std::size_t
  index_full_table(const NodeID node, const BlockID block) const {
    return _offsets[node] + block;
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_full_table(const NodeID node, const BlockID block) const {
    return weighted_degree_to_full_table(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  weighted_degree_to_full_table(const NodeID node, const BlockID block) const {
    return static_cast<EdgeWeight>(
        __atomic_load_n(&_gain_cache[index_full_table(node, block)], __ATOMIC_RELAXED)
    );
  }

  [[nodiscard]] KAMINPAR_INLINE bool use_full_table(const NodeID node) const {
    return _offsets[node + 1] - _offsets[node] == _k;
  }

  //
  // Init (mixed)
  //

  void reset() {
    SCOPED_TIMER("Reset gain cache");

    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) {
      _gain_cache[i] = 0;
    });

    _sparse_buffer_ets.clear();
  }

  void recompute() {
    SCOPED_TIMER("Recompute gain cache");

    _graph->pfor_nodes([&](const NodeID u) {
      _weighted_degrees[u] = 0;

      if (use_full_table(u)) {
        _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
          const BlockID block_v = _p_graph->block(v);
          _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);

          _gain_cache[index_full_table(u, block_v)] += static_cast<UnsignedEdgeWeight>(weight);
        });
      } else {
        auto ht = create_hash_table(u);
        _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
          const BlockID block_v = _p_graph->block(v);
          _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
          ht.increase_by(block_v, static_cast<UnsignedEdgeWeight>(weight));
        });
      }
    });
  }

  const Context &_ctx;

  const Graph *_graph = nullptr;
  const PartitionedGraph *_p_graph = nullptr;

  NodeID _n = kInvalidNodeID;
  BlockID _k = kInvalidBlockID;

  StaticArray<EdgeID> _offsets;

  // Number of bits reserved in hash table cells to store the key (i.e., block ID) of the entry
  int _bits_for_key = 0;

  StaticArray<UnsignedEdgeWeight> _gain_cache;
  StaticArray<UnsignedEdgeWeight> _weighted_degrees;

  mutable tbb::enumerable_thread_specific<FastResetArray<EdgeWeight>> _sparse_buffer_ets{[&] {
    return FastResetArray<EdgeWeight>(_k);
  }};
};

template <typename Graph>
using NormalHashingGainCache = HashingGainCache<Graph, GenericDeltaGainCache, true>;

template <typename Graph>
using LargeKHashingGainCache = HashingGainCache<Graph, LargeKGenericDeltaGainCache, false>;

} // namespace kaminpar::shm
