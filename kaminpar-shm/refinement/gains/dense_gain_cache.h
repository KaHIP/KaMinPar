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
#include <type_traits>

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/degree_buckets.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class DenseDeltaGainCache;

template <bool iterate_exact_gains = false> class DenseGainCache {
  SET_DEBUG(true);

  using Self = DenseGainCache<iterate_exact_gains>;
  template <typename, typename> friend class DenseDeltaGainCache;

  // Abuse MSB bit in the _weighted_degrees[] array for locking
  constexpr static UnsignedEdgeWeight kWeightedDegreeLock =
      (static_cast<UnsignedEdgeWeight>(1) << (std::numeric_limits<UnsignedEdgeWeight>::digits - 1));
  constexpr static UnsignedEdgeWeight kWeightedDegreeMask = ~kWeightedDegreeLock;

public:
  template <typename DeltaPartitionedGraph>
  using DeltaCache = DenseDeltaGainCache<DeltaPartitionedGraph, Self>;

  // gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = false;

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

    // This is the first vertex for which the sparse gain cache is used
    _node_threshold = 0;

    // This is the first degree-bucket s.t. all vertices in this and subsequent buckets use the
    // sparse gain cache
    _bucket_threshold = 0;

    // For vertices with the dense gain cache (i.e., hash table), we use the MSB bits to store the
    // target blocks and the LSB bits to store the gain values: compute bit masks and shifts for
    // both values
    // Note: these masks are only used for vertices < _node_threshold
    _bits_for_gain = sizeof(UnsignedEdgeWeight) * 8 - math::ceil_log2(_k);
    _gain_mask = (1ul << _bits_for_gain) - 1;
    _block_mask = ~_gain_mask;
    DBG << "[DenseGainCache] Reserve " << _bits_for_gain << " of " << sizeof(UnsignedEdgeWeight) * 8
        << " bits for gain values, " << math::ceil_log2(_k) << " bits for block IDs";

    std::size_t gc_size = 0;

    if (p_graph.sorted()) {
      DBG << "[DenseGainCache] Graph was rearranged by degree buckets: using the mixed "
             "dense-sparse strategy";

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
                   (lowest_degree_in_bucket<NodeID>(_bucket_threshold + 1) - 1);
        _node_threshold += p_graph.bucket_size(_bucket_threshold);
      }
      std::fill(_cache_offsets.begin() + _bucket_threshold, _cache_offsets.end(), gc_size);
      std::fill(
          _bucket_offsets.begin() + _bucket_threshold, _bucket_offsets.end(), _node_threshold
      );
      gc_size += (p_graph.n() - _node_threshold) * _k;

      DBG << "[DenseGainCache] Initialized with degree threshold: " << degree_threshold
          << ", node threshold: " << _node_threshold << ", bucket threshold: " << _bucket_threshold;
    } else {
      DBG << "[DenseGainCache] Graph was *not* rearranged by degree buckets: using the sparse "
             "strategy only";
      gc_size = 1ul * _n * _k;
    }
    DBG << "[DenseGainCache] Computed gain cache size: " << gc_size << " entries, allocate "
        << (gc_size * sizeof(UnsignedEdgeWeight) / 1024 / 1024) << " Mib)";

    if (_gain_cache.size() < gc_size) {
      SCOPED_TIMER("Allocation");
      _gain_cache.resize(static_array::noinit, gc_size);
    }
    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      _weighted_degrees.resize(static_array::noinit, _n);
    }
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

      const EdgeWeight conn_to = conn(node, to);
      if (conn_to > 0) {
        lambda(to, [&] { return conn(node, to) - conn_from; });
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

      if (v > _node_threshold) {
        __atomic_fetch_sub(&_gain_cache[index(v, block_from)], weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_gain_cache[index(v, block_to)], weight, __ATOMIC_RELAXED);
      } else {
        lock(v);
        // Decrease from weight
        const std::size_t idx_from = index(v, block_from);
        KASSERT(decode_entry_block(_gain_cache[idx_from]) == block_from);
        if (decode_entry_gain(__atomic_sub_fetch(&_gain_cache[idx_from], weight, __ATOMIC_RELAXED)
            ) == 0) {
          // erase
        }

        // Increase to weight
        const std::size_t idx_to = index(v, block_to);
        if (decode_entry_block(_gain_cache[idx_to]) != block_to) {
          KASSERT(decode_entry_block(_gain_cache[idx_to]) == 0);
          __atomic_fetch_store_n(
              &_gain_cache[idx_to], encode_cache_entry(block_to, 0), __ATOMIC_RELAXED
          );
        }
        __atomic_fetch_add(&_gain_cache[idx_to], weight, __ATOMIC_RELAXED);
        unlock(v);
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

private:
  void init_buckets(const Graph &graph) {
    _buckets.front() = 0;
    for (int bucket = 0; bucket < graph.number_of_buckets(); ++bucket) {
      _buckets[bucket + 1] = _buckets[bucket] + graph.bucket_size(bucket);
    }
    std::fill(_buckets.begin() + graph.number_of_buckets(), _buckets.end(), graph.n());
  }

  [[nodiscard]] int find_bucket(const NodeID node) const {
    int bucket = 0;
    while (node >= _buckets[bucket]) {
      for (int i = 0; i < 8; ++i) {
        bucket += (node >= _buckets[bucket]);
      }
    }
    return bucket;
  }

  BlockID decode_entry_block(const UnsignedEdgeWeight entry) {
    return static_cast<BlockID>((entry & _block_mask) >> _bits_for_gain);
  }

  EdgeWeight decode_entry_gain(const UnsignedEdgeWeight entry) {
    return static_cast<EdgeWeight>(entry & _gain_mask);
  }

  UnsignedEdgeWeight encode_cache_entry(const BlockID block, const EdgeWeight gain) {
    return (static_cast<UnsignedEdgeWeight>(block) << _bits_for_gain) |
           static_cast<UnsignedEdgeWeight>(gain);
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
        __ATOMIC_RELAXED,
        __ATOMIC_RELAXED
    ));
  }

  void unlock(const NodeID node) {
    __atomic_store_n(
        &_weighted_degrees[node], _weighted_degrees[node] & kWeightedDegreeMask, __ATOMIC_RELAXED
    );
  }

  [[nodiscard]] EdgeWeight weighted_degree(const NodeID node) const {
    KASSERT(node < _weighted_degrees[node]);
    return static_cast<EdgeWeight>(_weighted_degrees[node] & kWeightedDegreeMask);
  }

  [[nodiscard]] EdgeWeight weighted_degree_to(const NodeID node, const BlockID block) const {
    KASSERT(index(node, block) < _gain_cache.size());

    if (node > _node_threshold) {
      return static_cast<EdgeWeight>(
          __atomic_load_n(&_gain_cache[index(node, block)], __ATOMIC_RELAXED)
      );
    } else {
      return static_cast<EdgeWeight>(
          decode_entry_gain(__atomic_load_n(&_gain_cache[index(node, block)], __ATOMIC_RELAXED))
      );
    }
  }

  [[nodiscard]] std::size_t index(const NodeID node, const BlockID block) const {
    std::size_t idx;

    if (node > _node_threshold) {
      idx = _bucket_offsets[_bucket_threshold] +
            (static_cast<std::size_t>(node - _node_threshold) * static_cast<std::size_t>(_k) +
             static_cast<std::size_t>(block));
    } else {
      const auto bucket = find_bucket(node);
      const auto size = lowest_degree_in_bucket<NodeID>(bucket + 1);
      const auto mask = size - 1;

      const std::size_t offset = _cache_offsets[bucket] + (node - _buckets[bucket]) * size;
      std::size_t ht_pos = block;

      BlockID cur_block;
      do {
        ht_pos &= mask;
        cur_block = decode_entry_block(_gain_cache[offset + ht_pos]);
        ++ht_pos;
      } while (cur_block != block && cur_block != 0);
      --ht_pos;

      idx = offset + ht_pos;
    }

    KASSERT(idx < _gain_cache.size());
    return idx;
  }

  void reset() {
    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) {
      _gain_cache[i] = 0;
    });
  }

  void recompute_all(const PartitionedGraph &p_graph) {
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

      const std::size_t idx = index(u, block_v);
      if (u <= _node_threshold && decode_entry_block(_gain_cache[idx]) != block_v) {
        KASSERT(decode_entry_block(_gain_cache[idx]) == 0);
        _gain_cache[idx] = encode_cache_entry(block_v, 0);
      }
      _gain_cache[idx] += static_cast<UnsignedEdgeWeight>(weight);
      _weighted_degrees[u] += static_cast<UnsignedEdgeWeight>(weight);
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
  int _bits_for_gain = 0;

  StaticArray<UnsignedEdgeWeight> _gain_cache;
  StaticArray<UnsignedEdgeWeight> _weighted_degrees;
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
