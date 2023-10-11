/*******************************************************************************
 * On-the-fly gain calculator.
 *
 * @file:   gain_calculator.h
 * @author: Daniel Seemaier
 * @date:   03.07.2023
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {
template <bool randomize = true> class GainCalculator {
public:
  GainCalculator(const DistributedPartitionedGraph &p_graph) : _p_graph(p_graph) {}

  struct MaxGainer {
    EdgeWeight int_degree;
    EdgeWeight ext_degree;
    BlockID block;
    NodeWeight weight;

    EdgeWeight absolute_gain() const {
      return ext_degree - int_degree;
    }

    double relative_gain() const {
      if (ext_degree >= int_degree) {
        return 1.0 * absolute_gain() * weight;
      } else {
        return 1.0 * absolute_gain() / weight;
      }
    }
  };

  MaxGainer compute_max_gainer(const NodeID u, const PartitionContext &p_ctx) const {
    return compute_max_gainer_impl(
        u,
        [&p_ctx](const BlockID block, const BlockWeight weight_after_move) {
          return weight_after_move <= p_ctx.graph->max_block_weight(block);
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

private:
  template <typename WeightChecker>
  MaxGainer compute_max_gainer_impl(const NodeID u, WeightChecker &&weight_checker) const {
    const NodeWeight w_u = _p_graph.node_weight(u);
    const BlockID b_u = _p_graph.block(u);

    EdgeWeight int_degree = 0;
    EdgeWeight max_ext_degree = 0;
    BlockID target = b_u;

    Random &rand = Random::instance();

    auto action = [&](auto &map) {
      for (const auto [e, v] : _p_graph.neighbors(u)) {
        const BlockID b_v = _p_graph.block(v);
        if (b_u != b_v &&
            weight_checker(b_v, _p_graph.block_weight(b_v) + w_u)) {
          map[b_v] += _p_graph.edge_weight(e);
        } else if (b_u == b_v) {
          int_degree += _p_graph.edge_weight(e);
        }
      }

      for (const auto [block, gain] : map.entries()) {
        bool accept = gain > max_ext_degree;
        if constexpr (randomize) {
          accept = accept || (gain == max_ext_degree && rand.random_bool());
        }

        if (accept) {
          target = block;
          max_ext_degree = gain;
        }
      }

      map.clear();
    };

    auto &rating_map = _rating_map_ets.local();
    rating_map.run_with_map(action, action);

    // @todo Use a different hash map for low-degree vertices
    // @todo This requires us to use something other than FastResetArray<> as fallback map

    return {int_degree, max_ext_degree, target, w_u};
  }

  const DistributedPartitionedGraph &_p_graph;
  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight, BlockID>> _rating_map_ets{[&] {
    return RatingMap<EdgeWeight, BlockID>(_p_graph.k());
  }};
};

using RandomizedGainCalculator = GainCalculator<true>;
} // namespace kaminpar::dist
