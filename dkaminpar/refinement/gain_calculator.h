/*******************************************************************************
 * Gain calculator.
 *
 * @file:   gain_calculator.h
 * @author: Daniel Seemaier
 * @date:   03.07.2023
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/enumerable_thread_specific.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/dkaminpar.h"

#include "common/datastructures/rating_map.h"
#include "common/random.h"

namespace kaminpar::dist {
class GainCalculator {
public:
  GainCalculator(const DistributedPartitionedGraph &p_graph) : _p_graph(p_graph) {}

  struct MaxGainer {
    EdgeWeight int_degree;
    EdgeWeight max_ext_degree;
    BlockID max_gainer;
  };

  template <bool randomize = true>
  MaxGainer compute_max_gainer(const NodeID u, const PartitionContext &p_ctx) const {
    const NodeWeight u_weight = _p_graph.node_weight(u);
    const BlockID u_block = _p_graph.block(u);

    EdgeWeight int_degree = 0;
    EdgeWeight max_ext_degree = 0;
    BlockID max_gainer = u_block;

    Random &rand = Random::instance();

    auto action = [&](auto &map) {
      for (const auto [e, v] : _p_graph.neighbors(u)) {
        const BlockID v_block = _p_graph.block(v);
        if (u_block != v_block &&
            _p_graph.block_weight(v_block) + u_weight <= p_ctx.graph->max_block_weight(v_block)) {
          map[v_block] += _p_graph.edge_weight(e);
        } else if (u_block == v_block) {
          int_degree += _p_graph.edge_weight(e);
        }
      }

      for (const auto [block, gain] : map.entries()) {
        bool accept = gain > max_ext_degree;
        if constexpr (randomize) {
          accept = accept || (gain == max_ext_degree && rand.random_bool());
        }

        if (accept) {
          max_gainer = block;
          max_ext_degree = gain;
        }
      }

      map.clear();
    };

    auto &rating_map = _rating_map_ets.local();
    rating_map.run_with_map(action, action);

    // @todo Use a different hash map for low-degree vertices
    // @todo This requires us to use something other than FastResetArray<> as fallback map

    return {int_degree, max_ext_degree, max_gainer};
  }

  template <bool randomize = true>
  std::pair<EdgeWeight, BlockID>
  compute_absolute_gain(const NodeID u, const PartitionContext &p_ctx) const {
    const auto [int_degree, max_ext_degree, max_gainer] = compute_max_gainer<randomize>(u, p_ctx);
    const EdgeWeight gain = max_ext_degree - int_degree;
    return {gain, max_gainer};
  }

  template <bool randomize = true>
  std::pair<double, BlockID>
  compute_relative_gain(const NodeID u, const PartitionContext &p_ctx) const {
    const auto [absolute_gain, max_gainer] = compute_absolute_gain<randomize>(u, p_ctx);
    const NodeWeight weight = _p_graph.node_weight(u);
    if (absolute_gain >= 0) {
      return 1.0 * absolute_gain * weight;
    } else {
      return 1.0 * absolute_gain / weight;
    }
  }

private:
  const DistributedPartitionedGraph &_p_graph;

  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> _rating_map_ets{[&] {
    return RatingMap<EdgeWeight, BlockID>(_p_graph.k());
  }};
};
} // namespace kaminpar::dist
