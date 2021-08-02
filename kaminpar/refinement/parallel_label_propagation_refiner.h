/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "algorithm/parallel_label_propagation.h"
#include "refinement/i_refiner.h"
#include "utility/timer.h"

namespace kaminpar {
struct LabelPropagationRefinerConfig : public LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, SparseMap<NodeID, EdgeWeight>>;
  static constexpr bool kUseHardWeightConstraint = true;
  static constexpr bool kReportEmptyClusters = false;
};

class LabelPropagationRefiner final : public LabelPropagation<LabelPropagationRefiner, LabelPropagationRefinerConfig>,
                                      public Refiner {
  using Base = LabelPropagation<LabelPropagationRefiner, LabelPropagationRefinerConfig>;
  friend Base;

  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  LabelPropagationRefiner(const Graph &graph, const PartitionContext &p_ctx, const RefinementContext &r_ctx)
      : Base{graph.n(), p_ctx.k},
        _r_ctx{r_ctx} {
    set_max_degree(r_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(r_ctx.lp.max_num_neighbors);
  }

  [[nodiscard]] EdgeWeight expected_total_gain() const final { return Base::expected_total_gain(); }

  void initialize(const Graph &graph) final { _graph = &graph; }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final {
    ASSERT(_graph == &p_graph.graph());
    ASSERT(p_graph.k() <= p_ctx.k);
    _p_graph = &p_graph;
    _p_ctx = &p_ctx;
    Base::initialize(_graph); // we actually need _p_graph to initialize the algorithm

    const std::size_t max_iterations = _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER(TIMER_LABEL_PROPAGATION);
      const auto [num_moved_nodes, num_emptied_clusters] = randomized_iteration();
      if (num_moved_nodes == 0) { return false; }
    }

    return true;
  }

private:
  void reset_node_state(const NodeID) const {}
  [[nodiscard]] BlockID cluster(const NodeID u) const { return _p_graph->block(u); }
  void set_cluster(const NodeID u, const BlockID block) { _p_graph->set_block(u, block); }
  BlockID num_clusters() { return _p_graph->k(); }
  BlockWeight initial_cluster_weight(const BlockID block) { return _p_graph->block_weight(block); }
  BlockWeight max_cluster_weight(const BlockID block) { return _p_ctx->max_block_weight(block); }

  bool accept_cluster(const Base::ClusterSelectionState &state) {
    static_assert(std::is_signed_v<NodeWeight>);

    const NodeWeight current_max_weight = max_cluster_weight(state.current_cluster);
    const NodeWeight best_overload = state.best_cluster_weight - max_cluster_weight(state.best_cluster);
    const NodeWeight current_overload = state.current_cluster_weight - current_max_weight;
    const NodeWeight initial_overload = state.initial_cluster_weight - max_cluster_weight(state.initial_cluster);

    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain &&
             (current_overload < best_overload ||
              (current_overload == best_overload && state.local_rand.random_bool())))) &&
           (state.current_cluster_weight + state.u_weight < current_max_weight || current_overload < initial_overload ||
            state.current_cluster == state.initial_cluster);
  }

  const Graph *_graph{nullptr};
  PartitionedGraph *_p_graph{nullptr};
  const PartitionContext *_p_ctx;
  const RefinementContext &_r_ctx;
};
} // namespace kaminpar
