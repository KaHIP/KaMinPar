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
class OnTheFlyDeltaGainCache;

class OnTheFlyGainCache {
  friend OnTheFlyDeltaGainCache;

public:
  constexpr static bool kIteratesExactGains = true;

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

  template <typename TargetBlockAcceptor, typename GainConsumer>
  void gains(
      const NodeID node,
      const BlockID from,
      TargetBlockAcceptor &&target_block_acceptor,
      GainConsumer &&gain_consumer
  ) {
    gains_impl<PartitionedGraph>(
        *_p_graph,
        node,
        from,
        std::forward<TargetBlockAcceptor>(target_block_acceptor),
        std::forward<GainConsumer>(gain_consumer)
    );
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
    KASSERT(p_graph.block(node) == block);

    for (const auto [e, v] : p_graph.neighbors(node)) {
      if (p_graph.block(v) != block) {
        return true;
      }
    }

    return false;
  }
  template <typename PartitionedGraphType, typename TargetBlockAcceptor, typename GainConsumer>
  void gains_impl(
      const PartitionedGraphType &p_graph,
      const NodeID node,
      const BlockID from,
      TargetBlockAcceptor &&target_block_acceptor,
      GainConsumer &&gain_consumer
  ) {
    static_assert(std::is_invocable_r_v<bool, TargetBlockAcceptor, BlockID>);
    static_assert(std::is_invocable_r_v<void, GainConsumer, BlockID, EdgeWeight>);

    auto &rating_map = _rating_map_ets.local();
    rating_map.update_upper_bound_size(std::min<BlockID>(p_graph.degree(node), p_graph.k()));

    auto action = [&](auto &map) {
      for (const auto [e, v] : p_graph.neighbors(node)) {
        const BlockID block = p_graph.block(v);
        if (block == from || target_block_acceptor(block)) {
          map[block] += p_graph.edge_weight(e);
        }
      }
      const EdgeWeight conn_from = map[from];

      for (const auto [block, weight] : map.entries()) {
        if (block != conn_from) {
          gain_consumer(block, weight);
        }
      }
      map.clear();
    };
    rating_map.run_with_map(action, action);
  }

  const PartitionedGraph *_p_graph;

  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID, SparseMap<BlockID, EdgeWeight>>>
      _rating_map_ets;
};

class OnTheFlyDeltaGainCache {
public:
  constexpr static bool kIteratesExactGains = OnTheFlyGainCache::kIteratesExactGains;

  OnTheFlyDeltaGainCache(const OnTheFlyGainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(&d_graph) {}

  EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn_impl(*_d_graph, node, block);
  }

  EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain_impl(*_d_graph, node, from, to);
  }

  template <typename TargetBlockAcceptor, typename GainConsumer>
  void gains(
      const NodeID node,
      const BlockID from,
      TargetBlockAcceptor &&target_block_acceptor,
      GainConsumer &&gain_consumer
  ) const {
    _gain_cache.gains_impl(
        *_d_graph,
        node,
        from,
        std::forward<TargetBlockAcceptor>(target_block_acceptor),
        std::forward<GainConsumer>(gain_consumer)
    );
  }

  void move(
      const DeltaPartitionedGraph &d_graph, NodeID /* node */, BlockID /* from */, BlockID /* to */
  ) {
    // nothing to do
  }

  void clear() {}

private:
  const OnTheFlyGainCache &_gain_cache;
  const DeltaPartitionedGraph *_d_graph;
};

} // namespace kaminpar::shm
