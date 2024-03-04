/*******************************************************************************
 * Gain cache that uses dense cache for high degree nodes while computing gains
 * for low degree nodes on-the-fly.
 *
 * @file:   hybrid_high_degree_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   11.10.2023
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class HybridDeltaGainCache;

template <bool iterate_nonadjacent_blocks = true, bool iterate_exact_gains = true>
class HybridGainCache {
  SET_DEBUG(false);

  using Self = HybridGainCache<iterate_nonadjacent_blocks, iterate_exact_gains>;
  template <typename, typename> friend class HybridDeltaGainCache;

public:
  template <typename DeltaPartitionedGraph>
  using DeltaCache = HybridDeltaGainCache<DeltaPartitionedGraph, Self>;

  // gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;

  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block.
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  HybridGainCache(const Context &ctx, const NodeID preallocate_n, const BlockID preallocate_k)
      : _ctx(ctx),
        _on_the_fly_gain_cache(ctx, preallocate_n, preallocate_k),
        _gain_cache(static_array::noinit, 1ul * preallocate_n * preallocate_k),
        _weighted_degrees(static_array::noinit, preallocate_n) {}

  void initialize(const PartitionedGraph &p_graph) {
    DBG << "Initialize high-degree gain cache for a graph with n=" << p_graph.n()
        << ", k=" << p_graph.k();

    _n = p_graph.n();
    _k = p_graph.k();
    _high_degree_threshold = 0;

    if (p_graph.sorted()) {
      const EdgeID threshold = std::max<EdgeID>(
          _k * _ctx.refinement.kway_fm.k_based_high_degree_threshold,
          _ctx.refinement.kway_fm.constant_high_degree_threshold
      );

      for (int bucket = 0; _high_degree_threshold < p_graph.n() &&
                           p_graph.degree(_high_degree_threshold) < threshold;
           ++bucket) {
        _high_degree_threshold += p_graph.bucket_size(bucket);
      }

      _n = p_graph.n() - _high_degree_threshold;

      DBG << "Graph was rearranged: using the on-the-fly strategy for nodes with degree < " << _k
          << ": for all nodes up to " << _high_degree_threshold << ", i.e., we will keep " << _n
          << " nodes in the gain cache";
      if (_high_degree_threshold > 0) {
        DBG << "Last node not in the gain cache: " << _high_degree_threshold - 1 << " with degree "
            << p_graph.degree(_high_degree_threshold - 1);
      }
      if (_high_degree_threshold < p_graph.n()) {
        DBG << "First node in the gain cache: " << _high_degree_threshold << " with degree "
            << p_graph.degree(_high_degree_threshold);
      }
    } else {
      DBG << "Nodes are not sorted by degree buckets: using dense gain cache for all nodes";
    }

    // Mind the potential overflow without explicit cast
    const std::size_t gc_size = static_cast<std::size_t>(_n) * static_cast<std::size_t>(_k);

    DBG << "Allocating gain cache for n=" << _n << " nodes with k=" << _k
        << " blocks == " << gc_size << " slots";

    TIMED_SCOPE("Allocation") {
      _weighted_degrees.resize(static_array::noinit, _n);
      _gain_cache.resize(static_array::noinit, gc_size);
    };

    // Must initialize the on-the-fly gain cache before initializing the sparse part (i.e., calling
    // recompute_all()), otherwise the validation (only with heavy assertions) will not work
    _on_the_fly_gain_cache.initialize(p_graph);

    reset();
    recompute_all(p_graph);
  }

  void free() {
    tbb::parallel_invoke([&] { _gain_cache.free(); }, [&] { _weighted_degrees.free(); });
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    if (is_high_degree_node(node)) {
      return weighted_degree_to(node, to) - weighted_degree_to(node, from);
    } else {
      return _on_the_fly_gain_cache.gain(node, from, to);
    }
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) const {
    if (is_high_degree_node(node)) {
      return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
    } else {
      return _on_the_fly_gain_cache.gain(node, b_node, targets);
    }
  }

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    if (is_high_degree_node(node)) {
      return weighted_degree_to(node, block);
    } else {
      return _on_the_fly_gain_cache.conn(node, block);
    }
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    if (is_high_degree_node(node)) {
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
    } else {
      _on_the_fly_gain_cache.gains(node, from, std::forward<Lambda>(lambda));
    }
  }

  void
  move(const PartitionedGraph &p_graph, const NodeID node, const BlockID from, const BlockID to) {
    for (const auto &[e, v] : p_graph.neighbors(node)) {
      if (is_high_degree_node(v)) {
        const EdgeWeight w_e = p_graph.edge_weight(e);
        __atomic_fetch_sub(&_gain_cache[gc_index(v, from)], w_e, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_gain_cache[gc_index(v, to)], w_e, __ATOMIC_RELAXED);
      }
    }
  }

  [[nodiscard]] bool is_border_node(const NodeID node, const BlockID block) const {
    if (is_high_degree_node(node)) {
      return wd(node) != weighted_degree_to(node, block);
    } else {
      return _on_the_fly_gain_cache.is_border_node(node, block);
    }
  }

  [[nodiscard]] bool validate(const PartitionedGraph &p_graph) const {
    bool valid = true;
    p_graph.pfor_nodes([&](const NodeID u) {
      if (!dbg_check_cached_gain_for_node(p_graph, u)) {
        LOG_WARNING << "gain cache invalid for node " << u;
        std::exit(1); // @todo
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
    return __atomic_load_n(&gc(node, block), __ATOMIC_RELAXED);
  }

  [[nodiscard]] NodeWeight &wd(const NodeID node) {
    return _weighted_degrees[wd_index(node)];
  }

  [[nodiscard]] const NodeWeight &wd(const NodeID node) const {
    return _weighted_degrees[wd_index(node)];
  }

  [[nodiscard]] EdgeWeight &gc(const NodeID node, const BlockID block) {
    return _gain_cache[gc_index(node, block)];
  }

  [[nodiscard]] const EdgeWeight &gc(const NodeID node, const BlockID block) const {
    return _gain_cache[gc_index(node, block)];
  }

  [[nodiscard]] std::size_t wd_index(const NodeID node) const {
    KASSERT(is_high_degree_node(node));
    return node - _high_degree_threshold;
  }

  [[nodiscard]] std::size_t gc_index(const NodeID node, const BlockID block) const {
    KASSERT(is_high_degree_node(node));

    const std::size_t idx =
        static_cast<std::size_t>(node - _high_degree_threshold) * static_cast<std::size_t>(_k) +
        static_cast<std::size_t>(block);
    KASSERT(idx < _gain_cache.size());

    return idx;
  }

  void reset() {
    SCOPED_TIMER("Reset gain cache");
    tbb::parallel_for<std::size_t>(0, _gain_cache.size(), [&](const std::size_t i) {
      _gain_cache[i] = 0;
    });
  }

  void recompute_all(const PartitionedGraph &p_graph) {
    SCOPED_TIMER("Recompute gain cache");
    tbb::parallel_for<NodeID>(_high_degree_threshold, p_graph.n(), [&](const NodeID u) {
      recompute_node(p_graph, u);
    });
    KASSERT(
        validate(p_graph), "hybrid gain cache validation failed after recomputation", assert::heavy
    );
  }

  void recompute_node(const PartitionedGraph &p_graph, const NodeID u) {
    KASSERT(is_high_degree_node(u));
    KASSERT(u < p_graph.n());
    KASSERT(p_graph.block(u) < p_graph.k());

    const BlockID b_u = p_graph.block(u);
    wd(u) = 0;

    for (const auto &[e, v] : p_graph.neighbors(u)) {
      const EdgeWeight w_e = p_graph.edge_weight(e);
      gc(u, p_graph.block(v)) += w_e;
      wd(u) += w_e;
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
      if (actual_external_degrees[b] != conn(u, b)) {
        LOG_WARNING << "For " << (is_high_degree_node(u) ? "high-degree" : "low-degree") << " node "
                    << u << ": cached weighted degree to block " << b << " is " << conn(u, b)
                    << " but should be " << actual_external_degrees[b];
        return false;
      }
    }

    // There is no way to check this for nodes for which we use the on-the-fly gain cache
    if (is_high_degree_node(u) && actual_weighted_degree != wd(u)) {
      LOG_WARNING << "For high-degree node " << u << ": cached weighted degree is " << wd(u)
                  << " but should be " << actual_weighted_degree;
      return false;
    }

    return true;
  }

  [[nodiscard]] bool is_high_degree_node(const NodeID node) const {
    return node >= _high_degree_threshold;
  }

  const Context &_ctx;

  NodeID _high_degree_threshold = kInvalidNodeID;
  NodeID _n = kInvalidNodeID;
  BlockID _k = kInvalidBlockID;

  StaticArray<EdgeWeight> _gain_cache;
  StaticArray<EdgeWeight> _weighted_degrees;

  OnTheFlyGainCache<kIteratesNonadjacentBlocks, kIteratesExactGains> _on_the_fly_gain_cache;
};

template <typename _DeltaPartitionedGraph, typename _GainCache> class HybridDeltaGainCache {
public:
  using DeltaPartitionedGraph = _DeltaPartitionedGraph;
  using GainCache = _GainCache;

  // Delta gain caches can only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph.
  static_assert(GainCache::kIteratesNonadjacentBlocks);
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  HybridDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _on_the_fly_delta_gain_cache(_gain_cache._on_the_fly_gain_cache, d_graph) {}

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    if (_gain_cache.is_high_degree_node(node)) {
      return _gain_cache.conn(node, block) + conn_delta(node, block);
    } else {
      return _on_the_fly_delta_gain_cache.conn(node, block);
    }
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    if (_gain_cache.is_high_degree_node(node)) {
      return _gain_cache.gain(node, from, to) + conn_delta(node, to) - conn_delta(node, from);
    } else {
      return _on_the_fly_delta_gain_cache.gain(node, from, to);
    }
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) const {
    if (_gain_cache.is_high_degree_node(node)) {
      return {gain(node, b_node, targets.first), gain(node, b_node, targets.second)};
    } else {
      return _on_the_fly_delta_gain_cache.gain(node, b_node, targets);
    }
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    if (_gain_cache.is_high_degree_node(node)) {
      const EdgeWeight conn_from_delta = kIteratesExactGains ? conn_delta(node, from) : 0;

      _gain_cache.gains(node, from, [&](const BlockID to, auto &&gain) {
        lambda(to, [&] { return gain() + conn_delta(node, to) - conn_from_delta; });
      });
    } else {
      _on_the_fly_delta_gain_cache.gains(node, from, std::forward<Lambda>(lambda));
    }
  }

  void move(
      const DeltaPartitionedGraph &d_graph,
      const NodeID u,
      const BlockID block_from,
      const BlockID block_to
  ) {
    for (const auto &[e, v] : d_graph.neighbors(u)) {
      if (_gain_cache.is_high_degree_node(v)) {
        const EdgeWeight weight = d_graph.edge_weight(e);
        _gain_cache_delta[_gain_cache.gc_index(v, block_from)] -= weight;
        _gain_cache_delta[_gain_cache.gc_index(v, block_to)] += weight;
      }
    }
  }

  void clear() {
    _gain_cache_delta.clear();
    _on_the_fly_delta_gain_cache.clear();
  }

private:
  [[nodiscard]] EdgeWeight conn_delta(const NodeID node, const BlockID block) const {
    const auto it = _gain_cache_delta.get_if_contained(_gain_cache.gc_index(node, block));
    return it != _gain_cache_delta.end() ? *it : 0;
  }

  const GainCache &_gain_cache;
  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;

  OnTheFlyDeltaGainCache<
      DeltaPartitionedGraph,
      OnTheFlyGainCache<GainCache::kIteratesNonadjacentBlocks, GainCache::kIteratesExactGains>>
      _on_the_fly_delta_gain_cache;
};
} // namespace kaminpar::shm

