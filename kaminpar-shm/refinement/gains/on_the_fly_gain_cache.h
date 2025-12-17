/*******************************************************************************
 * Pseudo-gain cache that computes gains from scratch everytime they are needed.
 *
 * @file:   on_the_fly_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   28.09.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/inline.h"

namespace kaminpar::shm {

template <typename GainCache> class OnTheFlyDeltaGainCache;

template <
    typename GraphType,
    bool iterate_nonadjacent_blocks = true,
    bool iterate_exact_gains = true,
    bool iterate_source_block = false>
class OnTheFlyGainCache {
  template <typename> friend class OnTheFlyDeltaGainCache;

public:
  using Graph = GraphType;
  using Self = OnTheFlyGainCache<Graph, iterate_nonadjacent_blocks, iterate_exact_gains>;
  using DeltaGainCache = OnTheFlyDeltaGainCache<Self>;

  constexpr static bool kIteratesNonadjacentBlocks = iterate_nonadjacent_blocks;
  constexpr static bool kIteratesExactGains = iterate_exact_gains;
  constexpr static bool kIteratesSourceBlock = iterate_source_block;

  OnTheFlyGainCache(const Context & /* ctx */, NodeID /* max_n */, const BlockID preallocate_k)
      : _rating_map_ets([preallocate_k] {
          return RatingMap<EdgeWeight, BlockID, rm_backyard::SparseMap>(preallocate_k);
        }) {}

  void initialize(const Graph &graph, const PartitionedGraph &p_graph) {
    _graph = &graph;
    _p_graph = &p_graph;
    _k = p_graph.k();
  }

  void free() {
    // Nothing to do
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain(const NodeID node, const BlockID from, const BlockID to) const {
    return gain_impl(*_p_graph, node, from, to);
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) const {
    return gain_impl(*_p_graph, node, b_node, targets);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight conn(const NodeID node, const BlockID block) const {
    return conn_impl(*_p_graph, node, block);
  }

  void KAMINPAR_INLINE move(NodeID, BlockID, BlockID) {
    // Nothing to do
  }

  [[nodiscard]] KAMINPAR_INLINE bool is_border_node(const NodeID node, const BlockID block) const {
    return is_border_node_impl(*_p_graph, node, block);
  }

  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    gains_impl(*_p_graph, node, from, std::forward<Lambda>(lambda));
  }

  void print_statistics() const {
    // Nothing to do
  }

private:
  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain_impl(const auto &p_graph, const NodeID node, const BlockID from, const BlockID to) const {
    EdgeWeight conn_from = 0;
    EdgeWeight conn_to = 0;

    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      if (p_graph.block(v) == from) {
        conn_from += weight;
      } else if (p_graph.block(v) == to) {
        conn_to += weight;
      }
    });

    return conn_to - conn_from;
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight> gain_impl(
      const auto &p_graph,
      const NodeID node,
      const BlockID b_node,
      const std::pair<BlockID, BlockID> targets
  ) const {
    EdgeWeight conn_from = 0;
    std::pair<EdgeWeight, EdgeWeight> conns_to = {0, 0};

    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight w_e) {
      const BlockID b_v = p_graph.block(v);

      if (b_v == b_node) {
        conn_from += w_e;
      } else if (b_v == targets.first) {
        conns_to.first += w_e;
      } else if (b_v == targets.second) {
        conns_to.second += w_e;
      }
    });

    return {conns_to.first - conn_from, conns_to.second - conn_from};
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  conn_impl(const auto &p_graph, const NodeID node, const BlockID block) const {
    EdgeWeight conn = 0;

    _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
      if (p_graph.block(v) == block) {
        conn += weight;
      }
    });

    return conn;
  }

  [[nodiscard]] KAMINPAR_INLINE bool
  is_border_node_impl(const auto &p_graph, const NodeID node, const BlockID block) const {
    bool border_node = false;
    _graph->adjacent_nodes(node, [&](const NodeID v) {
      if (p_graph.block(v) != block) {
        border_node = true;
        return true;
      }

      return false;
    });

    return border_node;
  }

  template <typename Lambda>
  KAMINPAR_INLINE void
  gains_impl(const auto &p_graph, const NodeID node, const BlockID from, Lambda &&lambda) const {
    auto action = [&](auto &map) {
      _graph->adjacent_nodes(node, [&](const NodeID v, const EdgeWeight weight) {
        map[p_graph.block(v)] += weight;
      });
      const EdgeWeight conn_from = kIteratesExactGains ? map[from] : 0;

      if constexpr (kIteratesNonadjacentBlocks) {
        for (const BlockID to : p_graph.blocks()) {
          if (kIteratesSourceBlock || to != from) {
            lambda(to, [&] { return map[to] - conn_from; });
          }
        }
      } else {
        for (const auto [to, conn_to] : map.entries()) {
          if (kIteratesSourceBlock || to != from) {
            lambda(to, [&, conn_to = conn_to] { return conn_to - conn_from; });
          }
        }
      }

      map.clear();
    };

    if constexpr (kIteratesNonadjacentBlocks) {
      _rating_map_ets.local().execute(_k, action);
    } else {
      _rating_map_ets.local().execute(std::min<BlockID>(_graph->degree(node), _k), action);
    }
  }

  BlockID _k = kInvalidBlockID;

  const Graph *_graph = nullptr;
  const PartitionedGraph *_p_graph = nullptr;

  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID, rm_backyard::SparseMap>>
      _rating_map_ets;
};

template <typename GainCacheType> class OnTheFlyDeltaGainCache {
public:
  using GainCache = GainCacheType;

  // Delta gain caches can only be used with GainCaches that iterate over all blocks, since there
  // might be new connections to non-adjacent blocks in the delta graph.
  static_assert(GainCache::kIteratesNonadjacentBlocks);
  constexpr static bool kIteratesExactGains = GainCache::kIteratesExactGains;

  OnTheFlyDeltaGainCache(const GainCache &gain_cache, const DeltaPartitionedGraph &d_graph)
      : _gain_cache(gain_cache),
        _d_graph(d_graph) {}

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight conn(const NodeID node, const BlockID block) const {
    return _gain_cache.conn_impl(_d_graph, node, block);
  }

  [[nodiscard]] KAMINPAR_INLINE EdgeWeight
  gain(const NodeID node, const BlockID from, const BlockID to) const {
    return _gain_cache.gain_impl(_d_graph, node, from, to);
  }

  [[nodiscard]] KAMINPAR_INLINE std::pair<EdgeWeight, EdgeWeight>
  gain(const NodeID node, const BlockID b_node, const std::pair<BlockID, BlockID> &targets) const {
    return _gain_cache.gain_impl(_d_graph, node, b_node, targets);
  }

  template <typename Lambda>
  KAMINPAR_INLINE void gains(const NodeID node, const BlockID from, Lambda &&lambda) const {
    _gain_cache.gains_impl(_d_graph, node, from, std::forward<Lambda>(lambda));
  }

  KAMINPAR_INLINE void move(NodeID, BlockID, BlockID) {
    // Nothing to do
  }

  KAMINPAR_INLINE void clear() {
    // Nothing to do
  }

private:
  const GainCache &_gain_cache;
  const DeltaPartitionedGraph &_d_graph;
};

template <typename GraphType> using NormalOnTheFlyGainCache = OnTheFlyGainCache<GraphType>;

} // namespace kaminpar::shm
