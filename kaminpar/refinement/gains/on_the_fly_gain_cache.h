/*******************************************************************************
 * Pseudo-gain cache that computes gains from scratch everytime they are needed.
 *
 * @file:   on_the_fly_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   28.09.2023
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <kassert/kassert.hpp>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/datastructures/rating_map.h"
#include "common/datastructures/sparse_map.h"

namespace kaminpar::shm {
template <typename DeltaPartitionedGraph, typename GainCache> class OnTheFlyDeltaGainCache;

template <bool iterate_exact_gains = true> class OnTheFlyGainCache {
  using Self = OnTheFlyGainCache<iterate_exact_gains>;
  template <typename, typename> friend class OnTheFlyDeltaGainCache;

public:
  template <typename DeltaPartitionedGraph>
  using DeltaCache = OnTheFlyDeltaGainCache<DeltaPartitionedGraph, Self>;

  constexpr static bool kIteratesNonadjacentBlocks = false;
  constexpr static bool kIteratesExactGains = iterate_exact_gains;

  OnTheFlyGainCache(NodeID /* max_n */, BlockID max_k)
      : _rating_map_ets([&] {
          return RatingMap<EdgeWeight, BlockID, SparseMap<BlockID, EdgeWeight>>(max_k);
        }) {}

  void initialize(const PartitionedGraph &p_graph) {
    _p_graph = &p_graph;
  }

  void free() {
    // nothing to do
  }

  EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return gain_impl(*_p_graph, node, from, to);
  }

  EdgeWeight conn(const NodeID node, const BlockID block) const {
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

  bool is_border_node(const NodeID node, const BlockID block) const {
    return is_border_node_impl(*_p_graph, node, block);
  }

  bool validate(const PartitionedGraph & /* p_graph */) const {
    // nothing to do
    return true;
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    gains_impl<PartitionedGraph>(*_p_graph, node, from, std::forward<Lambda>(lambda));
  }

private:
  template <typename PartitionedGraphType>
  EdgeWeight gain_impl(
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
  EdgeWeight
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
  bool is_border_node_impl(
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
    auto &rating_map = _rating_map_ets.local();
    rating_map.update_upper_bound_size(std::min<BlockID>(p_graph.degree(node), p_graph.k()));

    auto action = [&](auto &map) {
      for (const auto [e, v] : p_graph.neighbors(node)) {
        map[p_graph.block(v)] += p_graph.edge_weight(e);
      }
      const EdgeWeight conn_from = kIteratesExactGains ? map[from] : 0;

      for (const auto [to, conn_to] : map.entries()) {
        if (to != from) {
          const EdgeWeight gain = conn_to - conn_from;
          lambda(to, [&] { return gain; });
        }
      }

      map.clear();
    };
    rating_map.run_with_map(action, action);
  }

  const PartitionedGraph *_p_graph;

  mutable tbb::enumerable_thread_specific<
      RatingMap<EdgeWeight, BlockID, SparseMap<BlockID, EdgeWeight>>>
      _rating_map_ets;
};

template <typename DeltaPartitionedGraph, typename GainCache> class OnTheFlyDeltaGainCache {
public:
  constexpr static bool kIteratesNonadjacentBlocks = GainCache::kIteratesNonadjacentBlocks;
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  OnTheFlyDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(&d_graph) {}

  EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn_impl(*_d_graph, node, block);
  }

  EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain_impl(*_d_graph, node, from, to);
  }

  template <typename Lambda>
  void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    static_assert(DeltaPartitionedGraph::kAllowsReadAfterMove, "illegal configuration");
    _gain_cache.gains_impl(*_d_graph, node, from, std::forward<Lambda>(lambda));
  }

  void move(
      const DeltaPartitionedGraph &d_graph, NodeID /* node */, BlockID /* from */, BlockID /* to */
  ) {
    // nothing to do
  }

  void clear() {}

private:
  const GainCache &_gain_cache;
  const DeltaPartitionedGraph *_d_graph;
};

} // namespace kaminpar::shm
