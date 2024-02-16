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

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {
template <bool randomize = true> class GainCalculator {
public:
  GainCalculator(const BlockID max_k)
      : _rating_map_ets([max_k] { return BlockRatingMap<EdgeWeight, BlockID>(max_k); }) {}

  struct MaxGainer {
    EdgeWeight int_degree;
    EdgeWeight ext_degree;
    BlockID block;
    NodeWeight weight;

    [[nodiscard]] EdgeWeight absolute_gain() const {
      return ext_degree - int_degree;
    }

    [[nodiscard]] double relative_gain() const {
      if (ext_degree >= int_degree) {
        return 1.0 * absolute_gain() * weight;
      } else {
        return 1.0 * absolute_gain() / weight;
      }
    }
  };

  void init(const DistributedPartitionedGraph &p_graph) {
    _p_graph = &p_graph;
  }

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
    KASSERT(_p_graph != nullptr, "GainCalculator not initialized!");

    Random &rand = Random::instance();

    const NodeWeight w_u = _p_graph->node_weight(u);
    const BlockID b_u = _p_graph->block(u);

    EdgeWeight int_conn = 0;
    EdgeWeight max_ext_conn = 0;
    BlockID max_target = b_u;

    auto action = [&](auto &map) {
      for (const auto [e, v] : _p_graph->neighbors(u)) {
        const BlockID b_v = _p_graph->block(v);
        if (b_u != b_v && weight_checker(b_v, _p_graph->block_weight(b_v) + w_u)) {
          map[b_v] += _p_graph->edge_weight(e);
        } else if (b_u == b_v) {
          int_conn += _p_graph->edge_weight(e);
        }
      }

      for (const auto [target, conn] : map.entries()) {
        if (conn > max_ext_conn || (randomize && conn == max_ext_conn && rand.random_bool())) {
          max_target = target;
          max_ext_conn = conn;
        }
      }

      map.clear();
    };

    _rating_map_ets.local().execute(std::min(_p_graph->k(), _p_graph->degree(u)), action);

    return {
        .int_degree = int_conn,
        .ext_degree = max_ext_conn,
        .block = max_target,
        .weight = w_u,
    };
  }

  const DistributedPartitionedGraph *_p_graph = nullptr;
  mutable tbb::enumerable_thread_specific<BlockRatingMap<EdgeWeight, BlockID>> _rating_map_ets;
};

using DeterministicGainCalculator = GainCalculator<false>;
using RandomizedGainCalculator = GainCalculator<true>;
} // namespace kaminpar::dist
