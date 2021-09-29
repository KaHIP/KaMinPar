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

#include "dkaminpar/coarsening/i_clustering_algorithm.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_context.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"
#include "kaminpar/datastructure/fast_reset_array.h"
#include "kaminpar/datastructure/rating_map.h"

#include <allocator/alignedallocator.hpp>
#include <data-structures/table_config.hpp>
#include <tbb/enumerable_thread_specific.h>
#include <utils/hash/murmur2_hash.hpp>

namespace dkaminpar {
template<typename ClusterID, typename ClusterWeight>
class OwnedRelaxedClusterWeightMap {
  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  using table_type = typename growt::table_config<ClusterID, ClusterWeight, hasher_type, allocator_type, hmod::growable,
                                                  hmod::deletion>::table_type;

public:
  explicit OwnedRelaxedClusterWeightMap(const ClusterID max_num_clusters) : _cluster_weights(max_num_clusters) {}

  auto &&take_cluster_weights() { return std::move(_cluster_weights); }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights.get_handle().insert(cluster, weight);
  }

  ClusterWeight cluster_weight(const ClusterID cluster) /* const */ {
    auto &handle = _cluster_weights_handles_ets.local();
    auto it = handle.find(cluster);
    ASSERT(it != handle.end());
    // ‘growt::migration_table_iterator<growt::migration_table_handle<growt::migration_table_data<growt::migration_table<growt::base_linear<growt::base_linear_config<growt::simple_slot<long unsigned int, int, true, 9223372036854775807>, utils_tm::hash_tm::murmur2_hash, growt::GenericAlignedAllocator<char, 128>, false, false, false> >, growt::table_config<long unsigned int, int, utils_tm::hash_tm::murmur2_hash, growt::GenericAlignedAllocator<char, 128>, hmod::growable, hmod::deletion>::workerstrat, growt::table_config<long unsigned int, int, utils_tm::hash_tm::murmur2_hash, growt::GenericAlignedAllocator<char, 128>, hmod::growable, hmod::deletion>::exclstrat> > >, false>’
//    static_cast<int>(it);
    // ‘growt::migration_table_reference<growt::migration_table_handle<growt::migration_table_data<growt::migration_table<growt::base_linear<growt::base_linear_config<growt::simple_slot<long unsigned int, int, true, 9223372036854775807>, utils_tm::hash_tm::murmur2_hash, growt::GenericAlignedAllocator<char, 128>, false, false, false> >, growt::table_config<long unsigned int, int, utils_tm::hash_tm::murmur2_hash, growt::GenericAlignedAllocator<char, 128>, hmod::growable, hmod::deletion>::workerstrat, growt::table_config<long unsigned int, int, utils_tm::hash_tm::murmur2_hash, growt::GenericAlignedAllocator<char, 128>, hmod::growable, hmod::deletion>::exclstrat> > >, false>’
//    typename decltype(*it)::value_type i = "asd";
    return (*it).second;
  }

  bool move_cluster_weight(const ClusterID old_cluster, const ClusterID new_cluster, const ClusterWeight delta,
                           const ClusterWeight max_weight) {
    if (cluster_weight(old_cluster) + delta <= max_weight) {
      auto &handle = _cluster_weights_handles_ets.local();

      const auto [old_it, old_found] = handle.update(
          old_cluster, [delta](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);
      const auto [new_it, new_found] = handle.update(
          new_cluster, [delta](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);

      ASSERT(old_found);
      UNUSED(old_found);
      ASSERT(new_found);
      UNUSED(new_found);

      return true;
    }
    return false;
  }

private:
  table_type _cluster_weights;
  tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
      [&] { return _cluster_weights.get_handle(); }};
};

struct DistributedGlobalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using ClusterID = GlobalNodeID;
  using ClusterWeight = GlobalNodeWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = true;
};

class DistributedGlobalLabelPropagationClustering final
    : public shm::InOrderLabelPropagation<DistributedGlobalLabelPropagationClustering,
                                          DistributedGlobalLabelPropagationClusteringConfig>,
      public OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>,
      public shm::OwnedClusterVector<NodeID, GlobalNodeID>,
      public ClusteringAlgorithm<GlobalNodeID> {
  SET_DEBUG(true);

  using Base = shm::InOrderLabelPropagation<DistributedGlobalLabelPropagationClustering,
                                            DistributedGlobalLabelPropagationClusteringConfig>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>;
  using ClusterBase = shm::OwnedClusterVector<NodeID, GlobalNodeID>;

  friend Base;

public:
  DistributedGlobalLabelPropagationClustering(const NodeID max_n, const CoarseningContext &c_ctx)
      : Base{max_n},
        ClusterWeightBase{max_n},
        ClusterBase{max_n} {
    set_max_num_iterations(c_ctx.lp.num_iterations);
    set_max_degree(c_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.lp.max_num_neighbors);
  }

  const AtomicClusterArray &compute_clustering(const DistributedGraph &graph,
                                               const NodeWeight max_cluster_weight) final;

  void set_max_num_iterations(const std::size_t max_num_iterations) {
    _max_num_iterations = max_num_iterations == 0 ? std::numeric_limits<std::size_t>::max() : max_num_iterations;
  }

public:
  [[nodiscard]] GlobalNodeID initial_cluster(const NodeID u) const { return _graph->local_to_global_node(u); }

  [[nodiscard]] NodeWeight initial_cluster_weight(const GlobalNodeID cluster) const {
    return _graph->node_weight(_graph->global_to_local_node(cluster));
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const GlobalNodeID /* cluster */) const { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) const { return _graph->is_owned_node(u); }

  using Base::_graph;
  NodeWeight _max_cluster_weight{std::numeric_limits<NodeWeight>::max()};
  std::size_t _max_num_iterations{std::numeric_limits<std::size_t>::max()};
};
} // namespace dkaminpar