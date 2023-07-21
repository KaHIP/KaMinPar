/*******************************************************************************
 * @file:   gain_cache.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 * @brief:
 ******************************************************************************/
#pragma once

#include <sparsehash/dense_hash_map>

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/datastructures/dynamic_map.h"
#include "common/logger.h"
#include "common/datastructures/noinit_vector.h"

namespace kaminpar::shm {
template <typename GainCache, bool use_sparsehash = false> class DeltaGainCache;

class DenseGainCache {
  friend class DeltaGainCache<DenseGainCache>;

public:
  DenseGainCache(const BlockID k, const NodeID n)
      : _k(k),
        _n(n),
        _gain_cache(static_cast<std::size_t>(_n) * static_cast<std::size_t>(_k)),
        _weighted_degrees(_n) {}

  void initialize(const PartitionedGraph &p_graph) {
    KASSERT(p_graph.k() <= _k, "gain cache is too small");
    KASSERT(p_graph.n() <= _n, "gain cache is too small");

    reset();
    recompute_all(p_graph);
  }

  EdgeWeight gain(const NodeID node, const BlockID block_from, const BlockID block_to) const {
    return weighted_degree_to(node, block_to) - weighted_degree_to(node, block_from);
  }

  EdgeWeight conn(const NodeID node, const BlockID block) const {
    return weighted_degree_to(node, block);
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

  bool is_border_node(const NodeID node, const BlockID block) const {
    KASSERT(node < _weighted_degrees.size());
    return _weighted_degrees[node] != weighted_degree_to(node, block);
  }

  bool validate(const PartitionedGraph &p_graph) const {
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
  EdgeWeight weighted_degree_to(const NodeID node, const BlockID block) const {
    KASSERT(index(node, block) < _gain_cache.size());
    return __atomic_load_n(&_gain_cache[index(node, block)], __ATOMIC_RELAXED);
  }

  std::size_t index(const NodeID node, const BlockID b) const {
    const std::size_t idx =
        static_cast<std::size_t>(node) * static_cast<std::size_t>(_k) + static_cast<std::size_t>(b);
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

      _gain_cache[index(u, block_v)] += weight;
      _weighted_degrees[u] += weight;
    }
  }

  bool check_cached_gain_for_node(const PartitionedGraph &p_graph, const NodeID u) const {
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

  BlockID _k;
  NodeID _n;

  NoinitVector<EdgeWeight> _gain_cache;
  NoinitVector<EdgeWeight> _weighted_degrees;
};

template <typename GainCache, bool use_sparsehash> class DeltaGainCache {
public:
  DeltaGainCache(const GainCache &gain_cache) : _gain_cache(gain_cache) {
    if constexpr (use_sparsehash) {
      _gain_cache_delta.set_empty_key(std::numeric_limits<std::size_t>::max());
    }
  }

  EdgeWeight conn(const NodeID node, const BlockID block) const {
    EdgeWeight delta = 0;
    if constexpr (use_sparsehash) {
      const auto it = _gain_cache_delta.find(_gain_cache.index(node, block));
      delta = it != _gain_cache_delta.end() ? it->second : 0;
    } else {
      const auto it = _gain_cache_delta.get_if_contained(_gain_cache.index(node, block));
      delta = it != _gain_cache_delta.end() ? *it : 0;
    }
    return _gain_cache.conn(node, block) + delta;
  }

  EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    EdgeWeight delta_to = 0;
    EdgeWeight delta_from = 0;

    if constexpr (use_sparsehash) {
      const auto it_to = _gain_cache_delta.find(_gain_cache.index(node, to));
      const auto it_from = _gain_cache_delta.find(_gain_cache.index(node, from));
      delta_to = it_to != _gain_cache_delta.end() ? it_to->second : 0;
      delta_from = it_from != _gain_cache_delta.end() ? it_from->second : 0;
    } else {
      const auto it_to = _gain_cache_delta.get_if_contained(_gain_cache.index(node, to));
      const auto it_from = _gain_cache_delta.get_if_contained(_gain_cache.index(node, from));
      delta_to = it_to != _gain_cache_delta.end() ? *it_to : 0;
      delta_from = it_from != _gain_cache_delta.end() ? *it_from : 0;
    }

    return _gain_cache.gain(node, from, to) + delta_to - delta_from;
  }

  template <typename DeltaPartitionedGraphType>
  void move(
      const DeltaPartitionedGraphType &d_graph,
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
  const GainCache &_gain_cache;

  std::conditional_t<
      use_sparsehash,
      google::dense_hash_map<std::size_t, EdgeWeight>,
      DynamicFlatMap<std::size_t, EdgeWeight>>
      _gain_cache_delta;
};
} // namespace kaminpar::shm
