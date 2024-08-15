/*******************************************************************************
 * @file:   hashing_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   13.08.2024
 ******************************************************************************/
#pragma once

#include <limits>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/compact_hash_map.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
template <
    typename GraphType,
    template <typename>
    typename DeltaGainCacheType,
    bool iterate_nonadjacent_blocks,
    bool iterate_exact_gains = false>
class HashingGainCache {
  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

  // Abuse MSB bit in the _weighted_degrees[] array for locking
  constexpr static UnsignedEdgeWeight kWeightedDegreeLock =
      (static_cast<UnsignedEdgeWeight>(1) << (std::numeric_limits<UnsignedEdgeWeight>::digits - 1));
  constexpr static UnsignedEdgeWeight kWeightedDegreeMask = ~kWeightedDegreeLock;

  struct Statistics {
    Statistics operator+(const Statistics &other) const {
      return {
          num_sparse_queries + other.num_sparse_queries,
          num_sparse_updates + other.num_sparse_updates,
          num_dense_queries + other.num_dense_queries,
          num_dense_updates + other.num_dense_updates,
          num_dense_insertions + other.num_dense_insertions,
          num_dense_deletions + other.num_dense_deletions,
          total_dense_fill_degree + other.total_dense_fill_degree,
          dense_fill_degree_count + other.dense_fill_degree_count,
          num_moves + other.num_moves,
      };
    }

    std::size_t num_sparse_queries = 0;
    std::size_t num_sparse_updates = 0;

    std::size_t num_dense_queries = 0;
    std::size_t num_dense_updates = 0;
    std::size_t num_dense_insertions = 0;
    std::size_t num_dense_deletions = 0;

    double total_dense_fill_degree = 0;
    std::size_t dense_fill_degree_count = 0;

    std::size_t num_moves = 0;
  };

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

  HashingGainCache(const Context &ctx, const NodeID preallocate_n, BlockID preallocate_k)
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

    if (_offset.size() < _n + 1) {
      SCOPED_TIMER("Allocation");
      _offset.resize(_n + 1);
    }

    _offset.front() = 0;
    _graph->pfor_nodes([&](const NodeID u) {
      _offset[u + 1] = std::min<EdgeID>(math::ceil2(_graph->degree(u)), _k);
    });
    parallel::prefix_sum(_offset.begin(), _offset.begin() + _n + 1, _offset.begin());
    const std::size_t gain_cache_size = _offset.back();

    if (_gain_cache.size() < gain_cache_size) {
      SCOPED_TIMER("Allocation");
      DBG << "Re-allocating dense gain cache to " << gain_cache_size * sizeof(EdgeWeight) / 1024
          << " KiB";
      _gain_cache.resize(gain_cache_size);
    }

    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(_n);
    }

    _bits_for_key = math::ceil_log2(_k);
    DBG << "Gain cache summary: have " << _n << " nodes, " << _k << " blocks";
    DBG << "  Reserve " << (sizeof(UnsignedEdgeWeight) * 8 - _bits_for_key) << " of "
        << sizeof(UnsignedEdgeWeight) * 8 << " bits for gain values";
    DBG << "  Reserve " << _bits_for_key << " of " << sizeof(UnsignedEdgeWeight) * 8
        << " bits for block IDs";

    reset();
    recompute_all();
  }

  void free() {
    tbb::parallel_invoke([&] { _gain_cache.free(); }, [&] { _weighted_degrees.free(); });
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
        auto &buffer = _dense_buffer_ets.local();

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
    IFSTATS(++_stats_ets.local().num_moves);

    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      if (use_full_table(v)) {
        __atomic_fetch_sub(&_gain_cache[index_full_table(v, block_from)], weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_gain_cache[index_full_table(v, block_to)], weight, __ATOMIC_RELAXED);

        IFSTATS(++_stats_ets.local().num_sparse_updates);
      } else {
        auto table = create_hash_table(v);

        lock(v);
        [[maybe_unused]] bool was_deleted = table.decrease_by(block_from, weight);
        [[maybe_unused]] bool was_inserted = table.increase_by(block_to, weight);
        unlock(v);

        IFSTATS(++_stats_ets.local().num_dense_updates);
        IFSTATS(_stats_ets.local().num_dense_deletions += (was_deleted ? 1 : 0));
        IFSTATS(_stats_ets.local().num_dense_insertions += (was_inserted ? 1 : 0));
      }
    });
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  is_border_node(const NodeID node, const BlockID block_of_node) const {
    return weighted_degree(node) != weighted_degree_to(node, block_of_node);
  }

  [[nodiscard]] bool validate() const {
    bool valid = true;
    _graph->pfor_nodes([&](const NodeID u) {
      if (!dbg_check_cached_gain_for_node(u)) {
        LOG_WARNING << "gain cache invalid for node " << u;
        valid = false;
      }
    });
    return valid;
  }

  void print_statistics() const {
    Statistics stats = _stats_ets.combine(std::plus{});

    LOG_STATS << "Sparse Gain Cache:";
    LOG_STATS << "  * # of moves: " << stats.num_moves;
    LOG_STATS << "  * # of queries: " << stats.num_dense_queries << " LD, "
              << stats.num_sparse_queries << " HD";
    LOG_STATS << "    + Average initial LD fill degree: "
              << (stats.dense_fill_degree_count > 0
                      ? 100.0 * stats.total_dense_fill_degree / stats.dense_fill_degree_count
                      : 0)
              << "%";
    LOG_STATS << "  * # of updates: " << stats.num_dense_updates << " LD, "
              << stats.num_sparse_updates << " HD";
    LOG_STATS << "    + # of LD Insertions: " << stats.num_dense_insertions;
    LOG_STATS << "    + # of LD Deletions: " << stats.num_dense_deletions;
  }

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
    IFSTATS(_stats_ets.local().num_sparse_queries += use_full_table(node));
    IFSTATS(_stats_ets.local().num_dense_queries += !use_full_table(node));

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
    IFSTATS(++_stats_ets.local().num_dense_queries);
    return static_cast<EdgeWeight>(create_hash_table(node).get(block));
  }

  [[nodiscard]] KAMINPAR_INLINE CompactHashMap<UnsignedEdgeWeight const, true>
  create_hash_table(const NodeID node) const {
    const std::size_t start = _offset[node];
    const std::size_t size = _offset[node + 1] - start;
    KASSERT(math::is_power_of_2(size));
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  [[nodiscard]] KAMINPAR_INLINE CompactHashMap<UnsignedEdgeWeight, true>
  create_hash_table(const NodeID node) {
    const std::size_t start = _offset[node];
    const std::size_t size = _offset[node + 1] - start;
    KASSERT(math::is_power_of_2(size));
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  //
  // Lookups (full table)
  //

  [[nodiscard]] KAMINPAR_INLINE std::size_t
  index_full_table(const NodeID node, const BlockID block) const {
    return _offset[node] + block;
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_full_table(const NodeID node, const BlockID block) const {
    return weighted_degree_to_full_table(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  weighted_degree_to_full_table(const NodeID node, const BlockID block) const {
    IFSTATS(++_stats_ets.local().num_sparse_queries);
    return static_cast<EdgeWeight>(
        __atomic_load_n(&_gain_cache[index_full_table(node, block)], __ATOMIC_RELAXED)
    );
  }

  [[nodiscard]] KAMINPAR_INLINE bool use_full_table(const NodeID node) const {
    return _offset[node + 1] - _offset[node] == _k;
  }

  //
  // Init (mixed)
  //

  void reset() {
    SCOPED_TIMER("Reset gain cache");
    IFSTATS(_stats_ets.clear());

    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) {
      _gain_cache[i] = 0;
    });
    _dense_buffer_ets.clear();
  }

  void recompute_all() {
    SCOPED_TIMER("Recompute gain cache");

    _graph->pfor_nodes([&](const NodeID u) { recompute_node(u); });
    KASSERT(validate(), "dense gain cache verification failed after recomputation", assert::heavy);

    IF_STATS {
      _graph->pfor_nodes([&](const NodeID u) {
        if (!use_full_table(u)) {
          auto map = create_hash_table(u);

          auto &stats = _stats_ets.local();
          stats.total_dense_fill_degree += 1.0 * map.count() / map.capacity();
          ++stats.dense_fill_degree_count;
        }
      });
    }
  }

  void recompute_node(const NodeID u) {
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
  }

  [[nodiscard]] bool dbg_check_cached_gain_for_node(const NodeID u) const {
    const BlockID block_u = _p_graph->block(u);
    std::vector<EdgeWeight> actual_external_degrees(_k, 0);
    EdgeWeight actual_weighted_degree = 0;

    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      const BlockID block_v = _p_graph->block(v);

      actual_weighted_degree += weight;
      actual_external_degrees[block_v] += weight;
    });

    for (BlockID b = 0; b < _k; ++b) {
      if (actual_external_degrees[b] != weighted_degree_to(u, b)) {
        LOG_WARNING << "For node " << u << ": cached weighted degree to block " << b << " is "
                    << weighted_degree_to(u, b) << " but should be " << actual_external_degrees[b];
        return false;
      }
    }

    if (actual_weighted_degree != weighted_degree(u)) {
      LOG_WARNING << "For node " << u << ": cached weighted degree is " << weighted_degree(u)
                  << " but should be " << actual_weighted_degree;
      return false;
    }

    return true;
  }

  const Context &_ctx;

  const Graph *_graph = nullptr;
  const PartitionedGraph *_p_graph = nullptr;

  NodeID _n = kInvalidNodeID;
  BlockID _k = kInvalidBlockID;

  StaticArray<EdgeID> _offset;

  // Number of bits reserved in hash table cells to store the key (i.e., block ID) of the entry
  int _bits_for_key = 0;

  std::size_t _sparse_offset = 0;

  StaticArray<UnsignedEdgeWeight> _gain_cache;
  StaticArray<UnsignedEdgeWeight> _weighted_degrees;

  mutable tbb::enumerable_thread_specific<Statistics> _stats_ets;
  mutable tbb::enumerable_thread_specific<FastResetArray<EdgeWeight>> _dense_buffer_ets{[&] {
    return FastResetArray<EdgeWeight>(_k);
  }};
};

template <typename Graph>
using NormalHashingGainCache = HashingGainCache<Graph, SparseDeltaGainCache, true>;

template <typename Graph>
using LargeKHashingGainCache = HashingGainCache<Graph, LargeKSparseDeltaGainCache, false>;
} // namespace kaminpar::shm
