/*******************************************************************************
 * Dense gain cache with O(m) memory -- *ONLY* for graphs that have been
 * rearranged by degree buckets. Fallbacks to the sparse gain cache with small
 * overheads otherwise.
 *
 * Gains for low-degree vertices (degree < k) are stored in a hash map using
 * linear probing. Modifying operations lock the hash-table, whereas read-only
 * operations are subject to race conditions.
 * Gains for high-degree vertices (degree >= k) are stored in the sparse part
 * of the array, i.e., with k entries per vertex.
 *
 * @ToDo: the hash tables store the target blocks + gain in the same 32 bit entry
 * (or 64 bit, when built with 64 bit edge weights). This is not ideal, especially
 * since the implementation exhibits undefined behaviour if this assumption does
 * not work out ...
 *
 * @file:   dense_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   07.02.2024
 ******************************************************************************/
#pragma once

#include <limits>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/compact_hash_map.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class DenseDeltaGainCache;

template <bool iterate_nonadjacent_blocks = true, bool iterate_exact_gains = false>
class DenseGainCache {
  SET_DEBUG(true);
  SET_STATISTICS(false);

  using Self = DenseGainCache<iterate_nonadjacent_blocks, iterate_exact_gains>;
  template <typename, typename> friend class DenseDeltaGainCache;

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
  template <typename DeltaPartitionedGraph>
  using DeltaCache = DenseDeltaGainCache<DeltaPartitionedGraph, Self>;

  // gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;

  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block.
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  DenseGainCache(const Context &ctx, const NodeID preallocate_n, BlockID preallocate_k)
      : _ctx(ctx),
        // Since we do not know the size of the gain cache in advance (depends on vertex degrees),
        // we cannot preallocate it
        _gain_cache(static_array::noinit, 0),
        _weighted_degrees(static_array::noinit, preallocate_n) {}

  void initialize(const PartitionedGraph &p_graph) {
    _n = p_graph.n();
    _k = p_graph.k();

    _node_threshold = 0;
    _bucket_threshold = 0;
    _cache_offsets[0] = 0;

    // Gains for nodes in the dense part of the gain cache are stored in small hash tables with
    // linear probing. Each slot of the hash table stores the right key (i.e., block) and its gain
    // value in the same 32 bit / 64 bit integer (depending on the size of the EdgeWeight data
    // type).
    // Thus, we compute the number of bits that we must reserve for the block IDs.
    _bits_for_key = math::ceil_log2(_k);
    DBG << "Reserve " << (sizeof(UnsignedEdgeWeight) * 8 - _bits_for_key) << " of "
        << sizeof(UnsignedEdgeWeight) * 8 << " bits for gain values, " << _bits_for_key
        << " bits for block IDs";

    // Size of the gain cache (dense + sparse part)
    std::size_t gc_size = 0;

    if (p_graph.sorted()) {
      DBG << "Graph was rearranged by degree buckets: using the mixed dense-sparse strategy";

      // Compute the degree that we use to determine the threshold degree bucket: nodes in buckets
      // up to the one determined by this degree are assigned to the dense part, the other ones to
      // the sparse part.
      const EdgeID degree_threshold = std::max<EdgeID>(
          _k * _ctx.refinement.kway_fm.k_based_high_degree_threshold, // usually k * 1
          _ctx.refinement.kway_fm.constant_high_degree_threshold      // usually 0
      );

      // (i) compute size of the dense part (== hash tables) ...
      for (_bucket_threshold = 0;
           _node_threshold < p_graph.n() && p_graph.degree(_node_threshold) < degree_threshold;
           ++_bucket_threshold) {
        _cache_offsets[_bucket_threshold] = gc_size;
        _node_threshold += p_graph.bucket_size(_bucket_threshold);
        gc_size += p_graph.bucket_size(_bucket_threshold) *
                   (lowest_degree_in_bucket<NodeID>(_bucket_threshold + 1));
      }
      std::fill(_cache_offsets.begin() + _bucket_threshold, _cache_offsets.end(), gc_size);

      // + ... (ii) size of the sparse part (table with k entries per node)
      gc_size += static_cast<std::size_t>(p_graph.n() - _node_threshold) * _k;

      DBG << "Initialized with degree threshold: " << degree_threshold
          << ", node threshold: " << _node_threshold << ", bucket threshold: " << _bucket_threshold;
      DBG << "Cache offsets: " << _cache_offsets;
    } else {
      // For graphs that do not have degree buckets, assign all nodes to the sparse part
      gc_size = 1ul * _n * _k;

      DBG << "Graph was *not* rearranged by degree buckets: using the sparse strategy only (i.e., "
             "using node threshold: "
          << _node_threshold << ", bucket threshold: " << _bucket_threshold << ")";
      DBG << "Cache offsets: " << _cache_offsets;
    }

    _sparse_offset = _cache_offsets[_bucket_threshold];

    DBG << "Computed gain cache size: " << gc_size << " entries, consumes "
        << (gc_size * sizeof(UnsignedEdgeWeight) / 1024.0) << " KiB memory";

    if (_gain_cache.size() < gc_size) {
      DBG << "Re-allocating dense gain cache to " << gc_size * sizeof(EdgeWeight) / 1024 << " KiB";
      SCOPED_TIMER("Allocation");
      _gain_cache.resize(gc_size);
    }

    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(_n);
    }

    init_buckets(p_graph.graph());
    reset();
    recompute_all(p_graph);
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
    if (in_sparse_part(node)) {
      const EdgeWeight conn_from = kIteratesExactGains ? conn_sparse(node, from) : 0;

      for (BlockID to = 0; to < _k; ++to) {
        if (from == to) {
          continue;
        }

        if constexpr (kIteratesNonadjacentBlocks) {
          lambda(to, [&] { return conn_sparse(node, to) - conn_from; });
        } else {
          const EdgeWeight conn_to = conn_sparse(node, to);
          if (conn_to > 0) {
            lambda(to, [&] { return conn_to - conn_from; });
          }
        }
      }
    } else {
      const EdgeWeight conn_from = kIteratesExactGains ? conn_dense(node, from) : 0;

      if constexpr (kIteratesNonadjacentBlocks) {
        auto &buffer = _dense_buffer_ets.local();

        create_dense_wrapper(node).for_each([&](const BlockID to, const EdgeWeight conn_to) {
          buffer.set(to, conn_to);
        });

        for (BlockID to = 0; to < _k; ++to) {
          if (from != to) {
            lambda(to, [&] { return buffer.get(to) - conn_from; });
          }
        }

        buffer.clear();
      } else {
        create_dense_wrapper(node).for_each([&](const BlockID to, const EdgeWeight conn_to) {
          if (to != from) {
            lambda(to, [&] { return conn_to - conn_from; });
          }
        });
      }
    }
  }

  KAMINPAR_INLINE void move(
      const PartitionedGraph &p_graph,
      const NodeID node,
      const BlockID block_from,
      const BlockID block_to
  ) {
    IFSTATS(++_stats_ets.local().num_moves);

    for (const auto &[e, v] : p_graph.neighbors(node)) {
      const EdgeWeight weight = p_graph.edge_weight(e);

      if (in_sparse_part(v)) {
        __atomic_fetch_sub(&_gain_cache[index_sparse(v, block_from)], weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_gain_cache[index_sparse(v, block_to)], weight, __ATOMIC_RELAXED);

        IFSTATS(++_stats_ets.local().num_sparse_updates);
      } else {
        auto table = create_dense_wrapper(v);

        lock(v);
        [[maybe_unused]] bool was_deleted = table.decrease_by(block_from, weight);
        [[maybe_unused]] bool was_inserted = table.increase_by(block_to, weight);
        unlock(v);

        IFSTATS(++_stats_ets.local().num_dense_updates);
        IFSTATS(_stats_ets.local().num_dense_deletions += (was_deleted ? 1 : 0));
        IFSTATS(_stats_ets.local().num_dense_insertions += (was_inserted ? 1 : 0));
      }
    }
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  is_border_node(const NodeID node, const BlockID block_of_node) const {
    return weighted_degree(node) != weighted_degree_to(node, block_of_node);
  }

  [[nodiscard]] bool validate(const PartitionedGraph &p_graph) const {
    bool valid = true;
    p_graph.pfor_nodes([&](const NodeID u) {
      if (!dbg_check_cached_gain_for_node(p_graph, u)) {
        LOG_WARNING << "gain cache invalid for node " << u;
        valid = false;
      }
    });
    return valid;
  }

  void print_statistics() const {
    Statistics stats = _stats_ets.combine(std::plus{});

    LOG_STATS << "Dense Gain Cache:";
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
  // Degree buckets
  //

  void init_buckets(const Graph &graph) {
    _buckets.front() = 0;
    for (int bucket = 0; bucket < graph.number_of_buckets(); ++bucket) {
      _buckets[bucket + 1] = _buckets[bucket] + graph.bucket_size(bucket);
    }
    std::fill(_buckets.begin() + graph.number_of_buckets(), _buckets.end(), graph.n());
  }

  [[nodiscard]] int find_bucket(const NodeID node) const {
    // @todo: which variation is best?
    // (a) linear scan over _buckets
    // (b) degree() + fast log2
    // (c) binary search on _buckets
    int bucket = 0;
    while (node >= _buckets[bucket + 1]) {
      for (int i = 0; i < 8; ++i) {
        bucket += (node >= _buckets[bucket + 1]);
      }
    }
    return bucket;
  }

  //
  // Locking (dense part)
  //

  void lock(const NodeID node) {
    UnsignedEdgeWeight state = __atomic_load_n(&_weighted_degrees[node], __ATOMIC_RELAXED);
    do {
      while (state & kWeightedDegreeLock) {
        state = __atomic_load_n(&_weighted_degrees[node], __ATOMIC_RELAXED);
      }
    } while (__atomic_compare_exchange_n(
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
    IFSTATS(_stats_ets.local().num_sparse_queries += in_sparse_part(node));
    IFSTATS(_stats_ets.local().num_dense_queries += !in_sparse_part(node));

    if (in_sparse_part(node)) {
      const std::size_t idx = index_sparse(node, block);
      return static_cast<EdgeWeight>(__atomic_load_n(&_gain_cache[idx], __ATOMIC_RELAXED));
    } else {
      return static_cast<EdgeWeight>(create_dense_wrapper(node).get(block));
    }
  }

  //
  // Lookups (dense part)
  //

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_dense(const NodeID node, const BlockID block) const {
    return weighted_degree_to_dense(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  weighted_degree_to_dense(const NodeID node, const BlockID block) const {
    IFSTATS(++_stats_ets.local().num_dense_queries);
    return static_cast<EdgeWeight>(create_dense_wrapper(node).get(block));
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<std::size_t, std::size_t> index_dense(const NodeID node
  ) const {
    const int bucket = find_bucket(node);
    const std::size_t size = lowest_degree_in_bucket<NodeID>(bucket + 1);
    KASSERT(math::is_power_of_2(size));
    return std::make_pair(_cache_offsets[bucket] + (node - _buckets[bucket]) * size, size);
  }

  [[nodiscard]] KAMINPAR_INLINE CompactHashMap<UnsignedEdgeWeight const>
  create_dense_wrapper(const NodeID node) const {
    const auto [start, size] = index_dense(node);
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  [[nodiscard]] KAMINPAR_INLINE CompactHashMap<UnsignedEdgeWeight>
  create_dense_wrapper(const NodeID node) {
    const auto [start, size] = index_dense(node);
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  //
  // Lookups (sparse part)
  //

  [[nodiscard]] KAMINPAR_INLINE std::size_t
  index_sparse(const NodeID node, const BlockID block) const {
    return _sparse_offset + 1ull * (node - _node_threshold) * _k + block;
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_sparse(const NodeID node, const BlockID block) const {
    return weighted_degree_to_sparse(node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  weighted_degree_to_sparse(const NodeID node, const BlockID block) const {
    IFSTATS(++_stats_ets.local().num_sparse_queries);
    return static_cast<EdgeWeight>(
        __atomic_load_n(&_gain_cache[index_sparse(node, block)], __ATOMIC_RELAXED)
    );
  }

  [[nodiscard]] KAMINPAR_INLINE bool in_sparse_part(const NodeID node) const {
    return node >= _node_threshold;
  }

  void reset() {
    SCOPED_TIMER("Reset gain cache");
    IFSTATS(_stats_ets.clear());

    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) {
      _gain_cache[i] = 0;
    });
    _dense_buffer_ets.clear();
  }

  void recompute_all(const PartitionedGraph &p_graph) {
    SCOPED_TIMER("Recompute gain cache");

    p_graph.pfor_nodes([&](const NodeID u) { recompute_node(p_graph, u); });
    KASSERT(
        validate(p_graph), "dense gain cache verification failed after recomputation", assert::heavy
    );

    IF_STATS {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (!in_sparse_part(u)) {
          auto map = create_dense_wrapper(u);

          auto &stats = _stats_ets.local();
          stats.total_dense_fill_degree += 1.0 * map.count() / map.capacity();
          ++stats.dense_fill_degree_count;
        }
      });
    }
  }

  void recompute_node(const PartitionedGraph &p_graph, const NodeID u) {
    KASSERT(u < p_graph.n());
    KASSERT(p_graph.block(u) < p_graph.k());

    _weighted_degrees[u] = 0;

    if (in_sparse_part(u)) {
      for (const auto &[e, v] : p_graph.neighbors(u)) {
        const BlockID block_v = p_graph.block(v);
        const EdgeWeight weight = p_graph.edge_weight(e);
        _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
        _gain_cache[index_sparse(u, block_v)] += static_cast<UnsignedEdgeWeight>(weight);
      }
    } else {
      auto ht = create_dense_wrapper(u);
      for (const auto &[e, v] : p_graph.neighbors(u)) {
        const BlockID block_v = p_graph.block(v);
        const EdgeWeight weight = p_graph.edge_weight(e);
        _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
        ht.increase_by(block_v, static_cast<UnsignedEdgeWeight>(weight));
      }
    }
  }

  [[nodiscard]] bool
  dbg_check_cached_gain_for_node(const PartitionedGraph &p_graph, const NodeID u) const {
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

    if (actual_weighted_degree != weighted_degree(u)) {
      LOG_WARNING << "For node " << u << ": cached weighted degree is " << weighted_degree(u)
                  << " but should be " << actual_weighted_degree;
      return false;
    }

    return true;
  }

  const Context &_ctx;

  NodeID _n = kInvalidNodeID;
  BlockID _k = kInvalidBlockID;

  // First node ID assigned to the sparse part of the gain cache
  NodeID _node_threshold = kInvalidNodeID;
  // First degree bucket assigned to the sparse part of the gain cache
  int _bucket_threshold = -1;
  // Copy of the degree buckets
  std::array<NodeID, kNumberOfDegreeBuckets<NodeID>> _buckets;
  // For each degree bucket, the offset for vertices in that bucket in the gain cache
  std::array<std::size_t, kNumberOfDegreeBuckets<NodeID>> _cache_offsets;
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

template <typename _DeltaPartitionedGraph, typename _GainCache> class DenseDeltaGainCache {
public:
  using DeltaPartitionedGraph = _DeltaPartitionedGraph;
  using GainCache = _GainCache;

  // Delta gain caches can only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph.
  static_assert(GainCache::kIteratesNonadjacentBlocks);
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  DenseDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _k(d_graph.k()),
        _gain_cache(gain_cache) {}

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

  KAMINPAR_INLINE void move(
      const DeltaPartitionedGraph &d_graph,
      const NodeID u,
      const BlockID block_from,
      const BlockID block_to
  ) {
    for (const auto &[e, v] : d_graph.neighbors(u)) {
      const EdgeWeight weight = d_graph.edge_weight(e);
      _gain_cache_delta[index(v, block_from)] -= weight;
      _gain_cache_delta[index(v, block_to)] += weight;
    }
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

  BlockID _k;
  const GainCache &_gain_cache;
  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;
};
} // namespace kaminpar::shm
