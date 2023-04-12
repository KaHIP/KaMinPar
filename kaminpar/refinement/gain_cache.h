/*******************************************************************************
 * @file:   gain_cache.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 * @brief:
 ******************************************************************************/
#pragma once

#include <sparsehash/dense_hash_map>

#include <tbb/parallel_for.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/noinit_vector.h"

namespace kaminpar::shm {
template <typename GainCache> class DeltaGainCache;

class DenseGainCache {
  friend class DeltaGainCache<DenseGainCache>;

public:
  DenseGainCache(const BlockID k, const NodeID n)
      : _k(k),
        _n(n),
        _gain_cache(_n * _k) {}

  void initialize(const PartitionedGraph &p_graph) {
    reset();
    recompute_all(p_graph);
  }

  EdgeWeight gain(NodeID node, BlockID block_From, BlockID to) const {
    return weighted_degree(node, to) - weighted_degree(node, block_From);
  }

  void move(
      const PartitionedGraph &p_graph,
      const NodeID u,
      const BlockID from,
      const BlockID to
  ) {
    for (const auto &[v, e] : p_graph.neighbors(u)) {
      const EdgeWeight weight = p_graph.edge_weight(e);
      if (p_graph.block(v) == from) {
        __atomic_fetch_sub(
            &_gain_cache[index(v, from)], weight, __ATOMIC_RELAXED
        );
      } else if (p_graph.block(v) == to) {
        __atomic_fetch_add(
            &_gain_cache[index(v, to)], weight, __ATOMIC_RELAXED
        );
      }
    }
  }

private:
  EdgeWeight weighted_degree(const NodeID node, const BlockID b) const {
    return __atomic_load_n(&_gain_cache[index(node, b)], __ATOMIC_RELAXED);
  }

  std::size_t index(const NodeID node, const BlockID b) const {
    return node * _k + b;
  }

  void reset() {
    tbb::parallel_for<std::size_t>(
        0, _gain_cache.size(), [&](const std::size_t i) { _gain_cache[i] = 0; }
    );
  }

  void recompute_all(const PartitionedGraph &p_graph) {
    p_graph.pfor_nodes([&](const NodeID u) { recompute_node(p_graph, u); });
  }

  void recompute_node(const PartitionedGraph &p_graph, const NodeID u) {
    for (const auto &[v, e] : p_graph.neighbors(u)) {
      const EdgeWeight weight = p_graph.edge_weight(e);
      _gain_cache[index(u, p_graph.block(v))] += weight;
    }
  }

  BlockID _k;
  NodeID _n;

  NoinitVector<EdgeWeight> _gain_cache;
};

template <typename GainCache> class DeltaGainCache {
public:
  DeltaGainCache(
      const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph
  )
      : _gain_cache(gain_cache) {
    _gain_cache_delta.set_empty_key(std::numeric_limits<std::size_t>::max());
    _gain_cache_delta.set_deleted_key(
        std::numeric_limits<std::size_t>::max() - 1
    );
  }

  EdgeWeight
  gain(const NodeID node, const BlockID from, const BlockID to) const {
    const auto it_to = _gain_cache_delta.find(_gain_cache.index(node, to));
    const EdgeWeight delta_to =
        it_to != _gain_cache_delta.end() ? it_to->second : 0;
    const auto it_from = _gain_cache_delta.find(_gain_cache.index(node, from));
    const EdgeWeight delta_from =
        it_from != _gain_cache_delta.end() ? it_from->second : 0;
    return _gain_cache.gain(node, from, to) + delta_to - delta_from;
  }

  void move(
      const DeltaPartitionedGraph &d_graph,
      const NodeID u,
      const BlockID from,
      const BlockID to
  ) {
    for (const auto &[v, e] : d_graph.neighbors(u)) {
      const EdgeWeight weight = d_graph.edge_weight(e);
      if (d_graph.block(v) == from) {
        _gain_cache_delta[_gain_cache.index(v, from)] -= weight;
      } else if (d_graph.block(v) == to) {
        _gain_cache_delta[_gain_cache.index(v, to)] += weight;
      }
    }
  }

private:
  const GainCache &_gain_cache;

  google::dense_hash_map<std::size_t, EdgeWeight> _gain_cache_delta;
};
} // namespace kaminpar::shm
