/*******************************************************************************
 * Pseudo-gain cache that computes gains from scratch everytime they are needed.
 *
 * @file:   on_the_fly_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   28.09.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/sparse_map.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class OnTheFlyDeltaGainCache;

template <bool iterate_nonadjacent_blocks = true, bool iterate_exact_gains = true>
class OnTheFlyGainCache {
  using Self = OnTheFlyGainCache<iterate_nonadjacent_blocks, iterate_exact_gains>;
  template <typename, typename> friend class OnTheFlyDeltaGainCache;

public:
  template <typename DeltaPartitionedGraph>
  using DeltaCache = OnTheFlyDeltaGainCache<DeltaPartitionedGraph, Self>;

  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  OnTheFlyGainCache(const Context & /* ctx */, NodeID /* max_n */, const BlockID preallocate_k)
      : _rating_map_ets([preallocate_k] {
          return RatingMap<EdgeWeight, BlockID, SparseMap<BlockID, EdgeWeight>>(preallocate_k);
        }) {}

  void initialize(const PartitionedGraph &p_graph) {
    _p_graph = &p_graph;
  }

  void free() {
    // nothing to do
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return gain_impl(*_p_graph, node, from, to);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) const {
    return gain_impl(*_p_graph, node, b_node, targets);
  }

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    return conn_impl(*_p_graph, node, block);
  }

  void move(
      const PartitionedGraph & /* p_graph */,
      NodeID /* node */,
      BlockID /* from */,
      BlockID /* to */
  ) {
    // nothing to do
  }

  [[nodiscard]] bool is_border_node(const NodeID node, const BlockID block) const {
    return is_border_node_impl(*_p_graph, node, block);
  }

  [[nodiscard]] bool validate(const PartitionedGraph & /* p_graph */) const {
    // nothing to do
    return true;
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    gains_impl<PartitionedGraph>(*_p_graph, node, from, std::forward<Lambda>(lambda));
  }

  void print_statistics() const {
    // print statistics
  }

private:
  template <typename PartitionedGraphType>
  [[nodiscard]] EdgeWeight gain_impl(
      const PartitionedGraphType &p_graph, const NodeID node, const BlockID from, const BlockID to
  ) const {
    EdgeWeight conn_from = 0;
    EdgeWeight conn_to = 0;

    for (const auto [e, v] : p_graph.neighbors(node)) {
      if (p_graph.block(v) == from) {
        conn_from += p_graph.edge_weight(e);
      } else if (p_graph.block(v) == to) {
        conn_to += p_graph.edge_weight(e);
      }
    }

    return conn_to - conn_from;
  }

  template <typename PartitionedGraphType>
  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight> gain_impl(
      const PartitionedGraphType &p_graph,
      const NodeID node,
      const BlockID b_node,
      const std::pair<BlockID, BlockID> targets
  ) const {
    EdgeWeight conn_from = 0;
    std::pair<EdgeWeight, EdgeWeight> conns_to = {0, 0};

    for (const auto [e, v] : p_graph.neighbors(node)) {
      const BlockID b_v = p_graph.block(v);
      const EdgeWeight w_e = p_graph.edge_weight(e);
      if (b_v == b_node) {
        conn_from += w_e;
      } else if (b_v == targets.first) {
        conns_to.first += w_e;
      } else if (b_v == targets.second) {
        conns_to.second += w_e;
      }
    }

    return {conns_to.first - conn_from, conns_to.second - conn_from};
  }

  template <typename PartitionedGraphType>
  [[nodiscard]] EdgeWeight
  conn_impl(const PartitionedGraphType &p_graph, const NodeID node, const BlockID block) const {
    EdgeWeight conn = 0;

    for (const auto [e, v] : p_graph.neighbors(node)) {
      if (p_graph.block(v) == block) {
        conn += p_graph.edge_weight(e);
      }
    }

    return conn;
  }

  template <typename PartitionedGraphType>
  [[nodiscard]] bool is_border_node_impl(
      const PartitionedGraphType &p_graph, const NodeID node, const BlockID block
  ) const {
    for (const auto [e, v] : p_graph.neighbors(node)) {
      if (p_graph.block(v) != block) {
        return true;
      }
    }

    return false;
  }

  template <typename PartitionedGraphType, typename Lambda>
  void gains_impl(
      const PartitionedGraphType &p_graph, const NodeID node, const BlockID from, Lambda &&lambda
  ) const {
    auto action = [&](auto &map) {
      for (const auto [e, v] : p_graph.neighbors(node)) {
        map[p_graph.block(v)] += p_graph.edge_weight(e);
      }
      const EdgeWeight conn_from = kIteratesExactGains ? map[from] : 0;

      if constexpr (kIteratesNonadjacentBlocks) {
        for (const BlockID to : p_graph.blocks()) {
          if (to != from) {
            lambda(to, [&] { return map[to] - conn_from; });
          }
        }
      } else {
        for (const auto [to, conn_to] : map.entries()) {
          if (to != from) {
            lambda(to, [&, conn_to = conn_to] { return conn_to - conn_from; });
          }
        }
      }

      map.clear();
    };

    if constexpr (kIteratesNonadjacentBlocks) {
      _rating_map_ets.local().execute(p_graph.k(), action);
    } else {
      _rating_map_ets.local().execute(std::min<BlockID>(p_graph.degree(node), p_graph.k()), action);
    }
  }

  const PartitionedGraph *_p_graph = nullptr;

  mutable tbb::enumerable_thread_specific<
      RatingMap<EdgeWeight, BlockID, SparseMap<BlockID, EdgeWeight>>>
      _rating_map_ets;
};

template <typename _DeltaPartitionedGraph, typename _GainCache> class OnTheFlyDeltaGainCache {
public:
  using DeltaPartitionedGraph = _DeltaPartitionedGraph;
  using GainCache = _GainCache;

  // Delta gain caches can only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph.
  static_assert(GainCache::kIteratesNonadjacentBlocks);
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  OnTheFlyDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(d_graph) {}

  [[nodiscard]] EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn_impl(_d_graph, node, block);
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain_impl(_d_graph, node, from, to);
  }

  [[nodiscard]] std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) const {
    return _gain_cache.gain_impl(_d_graph, node, b_node, targets);
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    static_assert(DeltaPartitionedGraph::kAllowsReadAfterMove, "illegal configuration");
    _gain_cache.gains_impl(_d_graph, node, from, std::forward<Lambda>(lambda));
  }

  void move(
      const DeltaPartitionedGraph &d_graph, NodeID /* node */, BlockID /* from */, BlockID /* to */
  ) {
    // nothing to do
    KASSERT(&_d_graph == &d_graph, "move() called with bad delta graph");
  }

  void clear() {
    // nothing to do
  }

private:
  const GainCache &_gain_cache;
  const DeltaPartitionedGraph &_d_graph;
};
} // namespace kaminpar::shm
