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

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_context.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"
#include "kaminpar/datastructure/fast_reset_array.h"
#include "kaminpar/datastructure/rating_map.h"

namespace dkaminpar {
struct DistributedLocalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using ClusterID = NodeID;
  using ClusterWeight = NodeWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = true;
};

class DistributedLocalLabelPropagationClustering final
    : public shm::ChunkRandomizedLabelPropagation<DistributedLocalLabelPropagationClustering,
                                                  DistributedLocalLabelPropagationClusteringConfig>,
      public shm::OwnedClusterVector<NodeID, NodeID>,
      public shm::OwnedRelaxedClusterWeightVector<NodeID, NodeWeight> {
  SET_DEBUG(true);

  using Base = shm::ChunkRandomizedLabelPropagation<DistributedLocalLabelPropagationClustering,
                                                    DistributedLocalLabelPropagationClusteringConfig>;
  using ClusterBase = shm::OwnedClusterVector<NodeID, NodeID>;
  using ClusterWeightBase = shm::OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;

  static constexpr std::size_t kInfiniteIterations{std::numeric_limits<std::size_t>::max()};

public:
  using ClusterBase::cluster;
  using ClusterBase::init_cluster;
  using ClusterBase::move_node;
  using ClusterWeightBase::cluster_weight;
  using ClusterWeightBase::init_cluster_weight;
  using ClusterWeightBase::move_cluster_weight;

  DistributedLocalLabelPropagationClustering(const NodeID max_n, const LabelPropagationCoarseningContext &lp_ctx)
      : Base{max_n},
        ClusterBase{max_n},
        ClusterWeightBase{max_n},
        _max_cluster_weight{kInvalidBlockWeight} {
    set_max_degree(lp_ctx.large_degree_threshold);
    set_max_num_neighbors(lp_ctx.max_num_neighbors);
  }

  const auto &cluster(const DistributedGraph &graph, const NodeWeight max_cluster_weight,
                      const std::size_t max_iterations = kInfiniteIterations) {
    initialize(&graph, graph.n());
    _max_cluster_weight = max_cluster_weight;

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      if (perform_iteration() == 0) { break; }
    }

    return clusters();
  }

  //
  // Called from base class
  //

  [[nodiscard]] NodeID initial_cluster(const NodeID u) const { return u; }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) const { return _graph->node_weight(cluster); }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) const { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] bool accept_neighbor(const NodeID u) const { return _graph->is_owned_node(u); }

  [[nodiscard]] bool activate_neighbor(const NodeID u) const { return _graph->is_owned_node(u); }

  using Base::_graph;
  NodeWeight _max_cluster_weight;
};
} // namespace dkaminpar