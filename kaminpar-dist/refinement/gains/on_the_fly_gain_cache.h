/*******************************************************************************
 * On-the-fly gain calculator.
 *
 * @file:   on_the_fly_gain_cache.h
 * @author: Daniel Seemaier
 * @date:   03.07.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/refinement/gains/max_gainer.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {

template <typename Graph> class OnTheFlyGainCache {
public:
  constexpr static bool kRandomized = true;

  OnTheFlyGainCache(const Context &ctx)
      : _rating_map_ets([&] { return RatingMap<EdgeWeight, BlockID>(ctx.partition.k); }) {}

  void init(const Graph &graph, const DistributedPartitionedGraph &p_graph) {
    _p_graph = &p_graph;
    _graph = &graph;
  }

  void free() {
    // Nothing to do
  }

  void consolidate() {
    // Nothing to do
  }

  MaxGainer compute_max_gainer(const NodeID u, const PartitionContext &p_ctx) const {
    return compute_max_gainer_impl(
        u,
        [&p_ctx](const BlockID block, const BlockWeight weight_after_move) {
          return weight_after_move <= p_ctx.max_block_weight(block);
        }
    );
  }

  MaxGainer compute_max_gainer(const NodeID u, const BlockWeight max_block_weight) const {
    return compute_max_gainer_impl(
        u,
        [max_block_weight](BlockID /* block */, const BlockWeight weight_after_move) {
          return weight_after_move <= max_block_weight;
        }
    );
  }

  MaxGainer compute_max_gainer(const NodeID u) const {
    return compute_max_gainer_impl(u, [](BlockID /* block */, BlockWeight /* weight_after_move */) {
      return true;
    });
  }

  // Each set bit permits that target block. The caller must keep any capacity
  // snapshot represented by the mask valid throughout this read-only query.
  MaxGainer compute_max_gainer_for_blocks(const NodeID u, const std::uint64_t blocks) const {
    KASSERT(_p_graph != nullptr, "GainCalculator not initialized!");
    KASSERT(_p_graph->k() <= std::numeric_limits<std::uint64_t>::digits);
    return compute_max_gainer_impl<true>(u, [blocks](const BlockID block) {
      return (blocks & (std::uint64_t{1} << block)) != 0;
    });
  }

  void move(NodeID, BlockID, BlockID) {
    // Nothing to do
  }

private:
  template <bool kUseBlockMask = false, typename WeightChecker>
  MaxGainer compute_max_gainer_impl(const NodeID u, WeightChecker &&weight_checker) const {
    KASSERT(_p_graph != nullptr, "GainCalculator not initialized!");

    Random &rand = Random::instance();

    const NodeWeight w_u = _graph->node_weight(u);
    const BlockID b_u = _p_graph->block(u);

    EdgeWeight int_conn = 0;
    EdgeWeight max_ext_conn = 0;
    BlockID max_target = b_u;

    const auto is_allowed_block = [&](const BlockID block) {
      if constexpr (kUseBlockMask) {
        return weight_checker(block);
      } else {
        return weight_checker(block, _p_graph->block_weight(block) + w_u);
      }
    };

    auto action = [&](auto &map) {
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID b_v = _p_graph->block(v);
        if (b_u != b_v && is_allowed_block(b_v)) {
          map[b_v] += w;
        } else if (b_u == b_v) {
          int_conn += w;
        }
      });

      for (const auto [target, conn] : map.entries()) {
        if (conn > max_ext_conn || (kRandomized && conn == max_ext_conn && rand.random_bool())) {
          max_target = target;
          max_ext_conn = conn;
        }
      }

      map.clear();
    };

    _rating_map_ets.local().execute(
        std::min<std::size_t>(_p_graph->k(), _p_graph->degree(u)), action
    );

    return {
        .int_degree = int_conn,
        .ext_degree = max_ext_conn,
        .block = max_target,
        .weight = w_u,
    };
  }

  const DistributedPartitionedGraph *_p_graph = nullptr;
  const Graph *_graph = nullptr;
  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> _rating_map_ets;
};

} // namespace kaminpar::dist
