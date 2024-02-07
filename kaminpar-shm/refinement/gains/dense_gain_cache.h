/*******************************************************************************
 * Gain cache that uses at most O(m) memory.
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
      const Context &ctx, const NodeID preallocate_for_n, const BlockID preallocate_for_m
  )
      : _ctx(ctx),
        _gain_cache(static_array::noinit, 0),
        _weighted_degrees(static_array::noinit, preallocate_for_n) {}

  void initialize(const PartitionedGraph &p_graph) {
    _n = p_graph.n();
    _k = p_graph.k();
    _node_threshold = 0;
    _bucket_threshold = 0;

    std::size_t gc_size = 0;

    if (p_graph.sorted()) {
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

      DBG << "[FM] Graph was rearranged: using the dense strategy for nodes with degree < "
          << degree_threshold << " (= " << _node_threshold
          << " nodes), using the sparse strategy for the rest (= " << p_graph.n() - _node_threshold
          << " nodes)";
    } else {
      gc_size = 1ul * _n * _k;
      DBG << "[FM] Graph was not rearrange: using the sparse strategy for all nodes";
    }

    if (_gain_cache.size() < gc_size) {
      SCOPED_TIMER("Allocation");
      DBG << "[FM] Resizing dense gain cache to " << gc_size << " slots";
      _gain_cache.resize(static_array::noinit, gc_size);
    }
    if (_weighted_degrees.size() < _n) {
      SCOPED_TIMER("Allocation");
      DBG << "[FM] Resizing weighted degrees for " << _n << " nodes";
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
        // @todo
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

  [[nodiscard]] std::size_t hash(const NodeID node, const BlockID block) const {
    // @todo
    return 0;
  }

  [[nodiscard]] EdgeWeight weighted_degree(const NodeID node) const {
    KASSERT(node < _weighted_degrees[node]);
    return static_cast<EdgeWeight>(_weighted_degrees[node] & kWeightedDegreeMask);
  }

  [[nodiscard]] EdgeWeight weighted_degree_to(const NodeID node, const BlockID block) const {
    KASSERT(index(node, block) < _gain_cache.size());
    return static_cast<EdgeWeight>(
        __atomic_load_n(&_gain_cache[index(node, block)], __ATOMIC_RELAXED)
    );
  }

  [[nodiscard]] std::size_t index(const NodeID node, const BlockID block) const {
    std::size_t idx;

    if (node > _node_threshold) {
      idx = _bucket_offsets[_bucket_threshold] +
            (static_cast<std::size_t>(node - _node_threshold) * static_cast<std::size_t>(_k) +
             static_cast<std::size_t>(block));
    } else {
      // @todo
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

      _gain_cache[index(u, block_v)] += static_cast<UnsignedEdgeWeight>(weight);
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
  std::array<std::size_t, kNumberOfDegreeBuckets<NodeID>> _cache_offsets;
  std::array<NodeID, kNumberOfDegreeBuckets<NodeID>> _bucket_offsets;

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
