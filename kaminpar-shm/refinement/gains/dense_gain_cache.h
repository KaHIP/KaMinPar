/*******************************************************************************
 * Dense gain cache with O(m) memory -- *ONLY* for graphs that have been
 * rearranged by degree buckets. Fallbacks to the sparse gain cache with small
 * overheads otherwise.
 *
 * Gains for low-degree vertices (degree < k) are stored in a hash map using
 * linear probing. Modifying operations lock the hash-table, whereas read-only
 * operations are subject to race conditions.
 *
 * @ToDo: the hash tables store the target blocks + gain in the same 32 bit entry
 * (or 64 bit, when built with 64 bit edge weights). This is not ideal, especially
 * since the implementation exhibits undefined behaviour if this assumption does
 * not work out ...
 *
 * Gains for high-degree vertices (degree >= k) are stored in a sparse array,
 * i.e., with k entries per vertex.
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
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class DenseDeltaGainCache;

template <bool iterate_nonadjacent_blocks, bool iterate_exact_gains = false> class DenseGainCache {
  SET_DEBUG(true);
  SET_STATISTICS(true);

  using Self = DenseGainCache<iterate_nonadjacent_blocks, iterate_exact_gains>;
  template <typename, typename> friend class DenseDeltaGainCache;

  // Abuse MSB bit in the _weighted_degrees[] array for locking
  constexpr static UnsignedEdgeWeight kWeightedDegreeLock =
      (static_cast<UnsignedEdgeWeight>(1) << (std::numeric_limits<UnsignedEdgeWeight>::digits - 1));
  constexpr static UnsignedEdgeWeight kWeightedDegreeMask = ~kWeightedDegreeLock;

  struct Statistics {
    Statistics operator+(const Statistics &other) const {
      return {
          num_hd_queries + other.num_hd_queries,
          num_hd_updates + other.num_hd_updates,
          num_ld_queries + other.num_ld_queries,
          num_ld_updates + other.num_ld_updates,
          num_ld_insertions + other.num_ld_insertions,
          num_ld_deletions + other.num_ld_deletions,
          total_ld_fill_degree + other.total_ld_fill_degree,
          ld_fill_degree_count + other.ld_fill_degree_count,
          num_moves + other.num_moves,
      };
    }

    std::size_t num_hd_queries = 0;
    std::size_t num_hd_updates = 0;

    std::size_t num_ld_queries = 0;
    std::size_t num_ld_updates = 0;
    std::size_t num_ld_insertions = 0;
    std::size_t num_ld_deletions = 0;

    double total_ld_fill_degree = 0;
    std::size_t ld_fill_degree_count = 0;

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

  DenseGainCache(
      const Context &ctx, const NodeID preallocate_for_n, BlockID /* preallocate_for_k */
  )
      : _ctx(ctx),
        // Since we do not know the size of the gain cache in advance (depends on vertex degrees),
        // we cannot preallocate it
        _gain_cache(static_array::noinit, 0),
        _weighted_degrees(static_array::noinit, preallocate_for_n) {}

  void initialize(const PartitionedGraph &p_graph) {
    _n = p_graph.n();
    _k = p_graph.k();

    _node_threshold = 0;
    _bucket_threshold = 0;
    _cache_offsets[0] = 0;
    _bucket_offsets[0] = 0;

    // For vertices with the dense gain cache (i.e., hash table), we use the MSB bits to store the
    // target blocks and the LSB bits to store the gain values: compute bit masks and shifts for
    // both values
    // Note: these masks are only used for vertices < _node_threshold
    const int bits_for_gain = (sizeof(UnsignedEdgeWeight) * 8 - math::ceil_log2(_k));
    _bits_for_key = math::ceil_log2(_k);
    _gain_mask = (1ul << bits_for_gain) - 1;
    _block_mask = ~_gain_mask;
    DBG << "Reserve " << bits_for_gain << " of " << sizeof(UnsignedEdgeWeight) * 8
        << " bits for gain values, " << _bits_for_key << " bits for block IDs";

    std::size_t gc_size = 0;

    if (p_graph.sorted()) {
      DBG << "Graph was rearranged by degree buckets: using the mixed dense-sparse strategy";

      const EdgeID degree_threshold = std::max<EdgeID>(
          _k * _ctx.refinement.kway_fm.k_based_high_degree_threshold,
          _ctx.refinement.kway_fm.constant_high_degree_threshold
      );

      for (_bucket_threshold = 0;
           _node_threshold < p_graph.n() && p_graph.degree(_node_threshold) < degree_threshold;
           ++_bucket_threshold) {
        _cache_offsets[_bucket_threshold] = gc_size;
        _bucket_offsets[_bucket_threshold] = _node_threshold;

        gc_size += p_graph.bucket_size(_bucket_threshold) *
                   (lowest_degree_in_bucket<NodeID>(_bucket_threshold + 1));
        _node_threshold += p_graph.bucket_size(_bucket_threshold);
      }
      std::fill(_cache_offsets.begin() + _bucket_threshold, _cache_offsets.end(), gc_size);
      std::fill(
          _bucket_offsets.begin() + _bucket_threshold, _bucket_offsets.end(), _node_threshold
      );
      gc_size += (p_graph.n() - _node_threshold) * _k;

      DBG << "Initialized with degree threshold: " << degree_threshold
          << ", node threshold: " << _node_threshold << ", bucket threshold: " << _bucket_threshold;
      DBG << "Bucket offsets: " << _bucket_offsets;
      DBG << "Cache offsets: " << _cache_offsets;
    } else {
      DBG << "Graph was *not* rearranged by degree buckets: using the sparse strategy only";
      gc_size = 1ul * _n * _k;
    }
    DBG << "Computed gain cache size: " << gc_size << " entries, allocate "
        << (gc_size * sizeof(UnsignedEdgeWeight) / 1024.0) << " KiB";

    TIMED_SCOPE("Allocation") {
      _gain_cache.resize(static_array::noinit, gc_size);
      _weighted_degrees.resize(static_array::noinit, _n);
    };

    init_buckets(p_graph.graph());
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
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    const EdgeWeight conn_from = kIteratesExactGains ? conn(node, from) : 0;

    for (BlockID to = 0; to < _k; ++to) {
      if (from == to) {
        continue;
      }

      if constexpr (kIteratesNonadjacentBlocks) {
        lambda(to, [&] { return conn(node, to) - conn_from; });
      } else {
        const EdgeWeight conn_to = conn(node, to);
        if (conn_to > 0) {
          lambda(to, [&] { return conn_to - conn_from; });
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
    IFSTATS(++_stats_ets.local().num_moves);

    for (const auto &[e, v] : p_graph.neighbors(node)) {
      const EdgeWeight weight = p_graph.edge_weight(e);

      if (is_hd_node(v)) {
        __atomic_fetch_sub(&_gain_cache[hd_index(v, block_from)], weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_gain_cache[hd_index(v, block_to)], weight, __ATOMIC_RELAXED);

        IFSTATS(++_stats_ets.local().num_hd_updates);
      } else {
        auto ht = ld_ht(v);

        lock(v);
        [[maybe_unused]] bool was_deleted = ht.decrease_by(block_from, weight);
        [[maybe_unused]] bool was_inserted = ht.increase_by(block_to, weight);
        unlock(v);

        IFSTATS(++_stats_ets.local().num_ld_updates);
        IFSTATS(_stats_ets.local().num_ld_deletions += (was_deleted ? 1 : 0));
        IFSTATS(_stats_ets.local().num_ld_insertions += (was_inserted ? 1 : 0));
      }
    }
  }

  [[nodiscard]] bool is_border_node(const NodeID node, const BlockID block_of_node) const {
    return weighted_degree(node) != weighted_degree_to(node, block_of_node);
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

  void summarize() const {
    IF_STATS {
      Statistics stats = _stats_ets.combine(std::plus{});
      STATS << "[Statistics] DenseGainCache:";
      STATS << "[Statistics]   * Moves: " << stats.num_moves;
      STATS << "[Statistics]   * Queries: " << stats.num_ld_queries << " LD, "
            << stats.num_hd_queries << " HD";
      STATS << "[Statistics]     + Average initial LD fill degree: "
            << (stats.ld_fill_degree_count > 0
                    ? 100.0 * stats.total_ld_fill_degree / stats.ld_fill_degree_count
                    : 0)
            << "%";
      STATS << "[Statistics]   * Updates: " << stats.num_ld_updates << " LD, "
            << stats.num_hd_updates << " HD";
      STATS << "[Statistics]     + LD Insertions: " << stats.num_ld_insertions;
      STATS << "[Statistics]     + LD Deletions: " << stats.num_ld_deletions;
    }
  }

private:
  void init_buckets(const Graph &graph) {
    _buckets.front() = 0;
    for (int bucket = 0; bucket < graph.number_of_buckets(); ++bucket) {
      _buckets[bucket + 1] = _buckets[bucket] + graph.bucket_size(bucket);
    }
    std::fill(_buckets.begin() + graph.number_of_buckets(), _buckets.end(), graph.n());
    DBG << "Initialized buckets: " << _buckets;
  }

  [[nodiscard]] int find_bucket(const NodeID node) const {
    int bucket = 0;
    while (node >= _buckets[bucket + 1]) {
      for (int i = 0; i < 8; ++i) {
        bucket += (node >= _buckets[bucket + 1]);
      }
    }
    return bucket;
  }

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
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST
    ));
  }

  void unlock(const NodeID node) {
    __atomic_store_n(
        &_weighted_degrees[node], _weighted_degrees[node] & kWeightedDegreeMask, __ATOMIC_RELAXED
    );
  }

  [[nodiscard]] EdgeWeight weighted_degree(const NodeID node) const {
    KASSERT(node < _weighted_degrees.size());
    return static_cast<EdgeWeight>(_weighted_degrees[node] & kWeightedDegreeMask);
  }

  [[nodiscard]] EdgeWeight weighted_degree_to(const NodeID node, const BlockID block) const {
    if (is_hd_node(node)) {
      IFSTATS(++_stats_ets.local().num_hd_queries);
      return static_cast<EdgeWeight>(
          __atomic_load_n(&_gain_cache[hd_index(node, block)], __ATOMIC_RELAXED)
      );
    } else {
      IFSTATS(++_stats_ets.local().num_ld_queries);
      return ld_ht(node).get(block);
    }
  }

  [[nodiscard]] std::pair<std::size_t, std::size_t> bucket_start_size(const NodeID node) const {
    if (is_hd_node(node)) {
      return std::make_pair(
          _cache_offsets[_bucket_threshold] +
              static_cast<std::size_t>(node - _node_threshold) * static_cast<std::size_t>(_k),
          _k
      );
    } else {
      const int bucket = find_bucket(node);
      const std::size_t size = lowest_degree_in_bucket<NodeID>(bucket + 1);
      KASSERT(math::is_power_of_2(size));
      return std::make_pair(_cache_offsets[bucket] + (node - _buckets[bucket]) * size, size);
    }
  }

  [[nodiscard]] bool is_hd_node(const NodeID node) const {
    return node >= _node_threshold;
  }

  [[nodiscard]] std::size_t hd_index(const NodeID node, const BlockID block) const {
    return bucket_start_size(node).first + static_cast<std::size_t>(block);
  }

  [[nodiscard]] std::size_t d_index(const NodeID node, const BlockID block) const {
    return static_cast<std::size_t>(node) * _k + block;
  }

  [[nodiscard]] CompactHashMap<UnsignedEdgeWeight const> ld_ht(const NodeID node) const {
    const auto [start, size] = bucket_start_size(node);
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  [[nodiscard]] CompactHashMap<UnsignedEdgeWeight> ld_ht(const NodeID node) {
    const auto [start, size] = bucket_start_size(node);
    return {_gain_cache.data() + start, size, _bits_for_key};
  }

  void reset() {
    IFSTATS(_stats_ets.clear());

    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) {
      _gain_cache[i] = 0;
    });
  }

  void recompute_all(const PartitionedGraph &p_graph) {
    p_graph.pfor_nodes([&](const NodeID u) { recompute_node(p_graph, u); });
    KASSERT(
        validate(p_graph), "dense gain cache verification failed after recomputation", assert::heavy
    );
    IF_STATS {
      p_graph.pfor_nodes([&](const NodeID u) {
        if (!is_hd_node(u)) {
          auto map = ld_ht(u);

          auto &stats = _stats_ets.local();
          stats.total_ld_fill_degree += 1.0 * map.count() / map.capacity();
          ++stats.ld_fill_degree_count;
        }
      });
    }
  }

  void recompute_node(const PartitionedGraph &p_graph, const NodeID u) {
    KASSERT(u < p_graph.n());
    KASSERT(p_graph.block(u) < p_graph.k());

    _weighted_degrees[u] = 0;

    if (is_hd_node(u)) {
      for (const auto &[e, v] : p_graph.neighbors(u)) {
        const BlockID block_v = p_graph.block(v);
        const EdgeWeight weight = p_graph.edge_weight(e);
        _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
        _gain_cache[hd_index(u, block_v)] += static_cast<UnsignedEdgeWeight>(weight);
      }
    } else {
      auto ht = ld_ht(u);

      for (const auto &[e, v] : p_graph.neighbors(u)) {
        const BlockID block_v = p_graph.block(v);
        const EdgeWeight weight = p_graph.edge_weight(e);
        _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
        ht.increase_by(block_v, static_cast<UnsignedEdgeWeight>(weight));
      }
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

  NodeID _node_threshold = kInvalidNodeID;
  int _bucket_threshold = -1;
  std::array<NodeID, kNumberOfDegreeBuckets<NodeID>> _buckets;
  std::array<std::size_t, kNumberOfDegreeBuckets<NodeID>> _cache_offsets;
  std::array<NodeID, kNumberOfDegreeBuckets<NodeID>> _bucket_offsets;

  UnsignedEdgeWeight _gain_mask = 0;
  UnsignedEdgeWeight _block_mask = 0;
  int _bits_for_key = 0;

  StaticArray<UnsignedEdgeWeight> _gain_cache;
  StaticArray<UnsignedEdgeWeight> _weighted_degrees;

  mutable tbb::enumerable_thread_specific<Statistics> _stats_ets;
};

template <typename DeltaPartitionedGraph, typename GainCache> class DenseDeltaGainCache {
public:
  constexpr static bool kIteratesNonadjacentBlocks = GainCache::kIteratesNonadjacentBlocks;
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  DenseDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph & /* d_graph */)
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
      _gain_cache_delta[_gain_cache.d_index(v, block_from)] -= weight;
      _gain_cache_delta[_gain_cache.d_index(v, block_to)] += weight;
    }
  }

  void clear() {
    _gain_cache_delta.clear();
  }

private:
  [[nodiscard]] EdgeWeight conn_delta(const NodeID node, const BlockID block) const {
    const auto it = _gain_cache_delta.get_if_contained(_gain_cache.d_index(node, block));
    return it != _gain_cache_delta.end() ? *it : 0;
  }

  const GainCache &_gain_cache;
  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;
};
} // namespace kaminpar::shm
