/*******************************************************************************
 * Gain cache that caches one gain for each node and block, using a total of
 * O(|V| * k) memory.
 *
 * @file:   dense_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/datastructures/dynamic_map.h"
#include "common/logger.h"
#include "common/timer.h"

namespace kaminpar::shm {
template <typename GainCache> class DenseDeltaGainCache;

class DenseGainCache {
  friend DenseDeltaGainCache<DenseGainCache>;

public:
  using DeltaCache = DenseDeltaGainCache<DenseGainCache>;

  // These can be set to either false or true, and the gain cache will adjust its behavior
  // accordingly:
  // If set to true, gains() will iterate over all blocks, including those not adjacent to the node.
  constexpr static bool kIteratesNonadjacentBlocks = true;
  // If set to true, gains() will call the gain consumer with exact gains; otherwise, it will call
  // the gain consumer with the total edge weight between the node and nodes in the specific block
  // (more expensive, but safes a call to gain() if the exact gain for the best block is needed).
  constexpr static bool kIteratesExactGains = false;

  DenseGainCache(const NodeID max_n, const BlockID max_k)
      : _max_n(max_n),
        _max_k(max_k),
        _gain_cache(
            static_array::noinit,
            static_cast<std::size_t>(_max_n) * static_cast<std::size_t>(_max_k)
        ),
        _weighted_degrees(static_array::noinit, _max_n) {}

  void initialize(const PartitionedGraph &p_graph) {
    KASSERT(p_graph.n() <= _max_n, "gain cache is too small");
    KASSERT(p_graph.k() <= _max_k, "gain cache is too small");

    _n = p_graph.n();
    _k = p_graph.k();

    START_TIMER("Reset");
    reset();
    STOP_TIMER();
    START_TIMER("Recompute");
    recompute_all(p_graph);
    STOP_TIMER();
  }

  void free() {
    tbb::parallel_invoke([&] { _gain_cache.free(); }, [&] { _weighted_degrees.free(); });
  }

  EdgeWeight gain(const NodeID node, const BlockID block_from, const BlockID block_to) const {
    return weighted_degree_to(node, block_to) - weighted_degree_to(node, block_from);
  }

  EdgeWeight conn(const NodeID node, const BlockID block) const {
    return weighted_degree_to(node, block);
  }

  template <typename TargetBlockAcceptor, typename GainConsumer>
  void gains(
      const NodeID node,
      const BlockID from,
      TargetBlockAcceptor &&target_block_acceptor,
      GainConsumer &&gain_consumer
  ) const {
    static_assert(std::is_invocable_r_v<bool, TargetBlockAcceptor, BlockID>);
    static_assert(std::is_invocable_r_v<void, GainConsumer, BlockID, EdgeWeight>);

    const EdgeWeight conn_from = kIteratesExactGains ? conn(node, from) : 0;

    for (BlockID to = 0; to < _k; ++to) {
      if (from != to && target_block_acceptor(to)) {
        const EdgeWeight conn_to = conn(node, to);

        if constexpr (!kIteratesNonadjacentBlocks) {
          if (conn_to == 0) {
            continue;
          }
        }

        if constexpr (kIteratesExactGains) {
          gain_consumer(to, conn_to - conn_from);
        } else {
          gain_consumer(to, conn_to);
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

  std::size_t index(const NodeID node, const BlockID block) const {
    const std::size_t idx = static_cast<std::size_t>(node) * static_cast<std::size_t>(_k) +
                            static_cast<std::size_t>(block);
    KASSERT(idx < _gain_cache.size());
    return idx;
  }

  void reset() {
    tbb::parallel_for<std::size_t>(0, _n * _k, [&](const std::size_t i) { _gain_cache[i] = 0; });
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

  NodeID _max_n;
  BlockID _max_k;

  NodeID _n;
  BlockID _k;

  StaticArray<EdgeWeight> _gain_cache;
  StaticArray<EdgeWeight> _weighted_degrees;
};

template <typename GainCache> class DenseDeltaGainCache {
public:
  constexpr static bool kIteratesNonadjacentBlocks = GainCache::kIteratesNonadjacentBlocks;
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  DenseDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph & /* d_graph */)
      : _gain_cache(gain_cache) {}

  EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn(node, block) + conn_delta(node, block);
  }

  EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain(node, from, to) + conn_delta(node, to) - conn_delta(node, from);
  }

  template <typename TargetBlockAcceptor, typename GainConsumer>
  void gains(
      const NodeID node,
      const BlockID from,
      TargetBlockAcceptor &&target_block_acceptor,
      GainConsumer &&gain_consumer
  ) const {
    const EdgeWeight conn_from_delta = kIteratesExactGains ? conn_delta(node, from) : 0;

    _gain_cache.gains(
        node,
        from,
        std::forward<TargetBlockAcceptor>(target_block_acceptor),
        [&](const BlockID to, const EdgeWeight conn_to) {
          const EdgeWeight real_conn_to = conn_to + conn_delta(node, to);

          if constexpr (!kIteratesNonadjacentBlocks) {
            if (real_conn_to == 0) {
              return;
            }
          }

          if constexpr (kIteratesExactGains) {
            gain_consumer(to, real_conn_to - conn_from_delta);
          } else {
            gain_consumer(to, real_conn_to);
          }
        }
    );
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
  EdgeWeight conn_delta(const NodeID node, const BlockID block) const {
    const auto it = _gain_cache_delta.get_if_contained(_gain_cache.index(node, block));
    return it != _gain_cache_delta.end() ? *it : 0;
  }

  const GainCache &_gain_cache;
  DynamicFlatMap<std::size_t, EdgeWeight> _gain_cache_delta;
};
} // namespace kaminpar::shm
