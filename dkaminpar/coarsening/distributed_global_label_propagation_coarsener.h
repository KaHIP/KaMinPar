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

#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <utils/hash/murmur2_hash.hpp>

namespace dkaminpar {
template<typename ClusterID, typename ClusterWeight>
class OwnedRelaxedClusterWeightMap {
  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  using table_type = typename growt::table_config<ClusterID, ClusterWeight, hasher_type, allocator_type, hmod::growable,
                                                  hmod::deletion>::table_type;

protected:
  explicit OwnedRelaxedClusterWeightMap(const ClusterID max_num_clusters) : _cluster_weights(max_num_clusters) {}

  auto &&take_cluster_weights() { return std::move(_cluster_weights); }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) { _cluster_weights[cluster] = weight; }

  ClusterWeight cluster_weight(const ClusterID cluster) const { return _cluster_weights[cluster]; }

  bool move_cluster_weight(const ClusterID old_cluster, const ClusterID new_cluster, const ClusterWeight delta,
                           const ClusterWeight max_weight) {
    if (_cluster_weights[new_cluster] + delta <= max_weight) {
      _cluster_weights[new_cluster] += delta;
      _cluster_weights[old_cluster] -= delta;
      return true;
    }
    return false;
  }

private:
  table_type _cluster_weights;
};

struct DistributedGlobalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using ClusterID = GlobalNodeID;
  using ClusterWeight = GlobalNodeWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = true;
};

class DistributedGlobalLabelPropagationClustering final
    : public shm::LabelPropagation<DistributedGlobalLabelPropagationClustering,
                                   DistributedGlobalLabelPropagationClusteringConfig>,
      public OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>,
      public shm::OwnedClusterVector<NodeID, GlobalNodeID> {
  SET_DEBUG(true);

  using Base = shm::LabelPropagation<DistributedGlobalLabelPropagationClustering,
                                     DistributedGlobalLabelPropagationClusteringConfig>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>;
  using ClusterBase = shm::OwnedClusterVector<NodeID, GlobalNodeID>;

  static constexpr std::size_t kInfiniteIterations{std::numeric_limits<std::size_t>::max()};

public:
  DistributedGlobalLabelPropagationClustering(const NodeID max_n, const double shrink_factor,
                                              const LabelPropagationCoarseningContext &lp_ctx)
      : Base{max_n},
        ClusterWeightBase{max_n},
        ClusterBase{max_n},
        _max_cluster_weight{kInvalidBlockWeight} {
    set_max_degree(lp_ctx.large_degree_threshold);
    set_max_num_neighbors(lp_ctx.max_num_neighbors);
  }

  const auto &cluster(const DistributedGraph &graph, const NodeWeight max_cluster_weight,
                      const std::size_t max_iterations = kInfiniteIterations) {
    ASSERT(_clustering.size() >= graph.n());

    initialize(&graph);
    reset_ghost_node_state();

    _max_cluster_weight = max_cluster_weight;
    _current_size = graph.n();
    _target_size = static_cast<NodeID>(_shrink_factor * _current_size);

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      const auto [num_moved_nodes, num_emptied_clusters] = randomized_iteration();
      _current_size -= num_emptied_clusters;
      if (num_moved_nodes == 0) { break; }
    }

    return _clustering;
  }

private:
  void reset_ghost_node_state() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID u) { reset_node_state(u); });
  }

  // used in Base class
  void reset_node_state(const NodeID u) {
    _clustering[u] = _graph->local_to_global_node(u);
    _favored_clustering[u] = _graph->local_to_global_node(u);
  }

  [[nodiscard]] GlobalNodeID cluster(const NodeID u) const { return _clustering[u]; }
  void set_cluster(const NodeID u, const GlobalNodeID cluster) { _clustering[u] = cluster; }
  void set_favored_cluster(const NodeID u, const GlobalNodeID cluster) { _favored_clustering[u] = cluster; }
  [[nodiscard]] GlobalNodeID num_clusters() const { return _graph->n(); }
  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) const { return _graph->node_weight(cluster); }
  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) const { return _max_cluster_weight; }

  // do not join clusters of ghost nodes
  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] bool should_stop(const NodeID num_emptied_clusters) const {
    return _current_size - num_emptied_clusters < _target_size;
  }

  NodeWeight _max_cluster_weight;
};
} // namespace dkaminpar