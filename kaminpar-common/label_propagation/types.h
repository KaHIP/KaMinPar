/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   types.h
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#include "kaminpar-common/random.h"

namespace kaminpar::lp {

enum class RatingMapStrategy {
  SINGLE_PHASE,
  TWO_PHASE,
  GROWING_HASH_TABLES,
};

enum class ActiveSetStrategy {
  NONE,
  GLOBAL,
  LOCAL,
};

enum class TieBreakingStrategy {
  GEOMETRIC,
  UNIFORM,
};

template <typename NodeID, typename ClusterID> struct Initialization {
  NodeID num_nodes;
  NodeID num_active_nodes;
  ClusterID num_clusters;
};

template <typename NodeID, typename ClusterID> struct Options {
  NodeID max_degree = std::numeric_limits<NodeID>::max();
  NodeID max_num_neighbors = std::numeric_limits<NodeID>::max();
  ClusterID desired_num_clusters = 0;
  RatingMapStrategy rating_map_strategy = RatingMapStrategy::SINGLE_PHASE;
  ActiveSetStrategy active_set_strategy = ActiveSetStrategy::NONE;
  TieBreakingStrategy tie_breaking_strategy = TieBreakingStrategy::GEOMETRIC;
  bool track_cluster_count = false;
  bool use_two_hop_clustering = false;
  bool use_actual_gain = false;
  bool relabel_before_second_phase = false;
  std::size_t rating_map_threshold = 10000;
};

template <typename NodeID, typename EdgeWeight> struct MoveCandidate {
  NodeID node;
  EdgeWeight gain;
};

template <typename NodeID, typename ClusterID, typename EdgeWeight> struct PassResult {
  NodeID processed_nodes = 0;
  NodeID moved_nodes = 0;
  ClusterID removed_clusters = 0;
  EdgeWeight expected_total_gain = 0;
};

template <typename NodeID, typename ClusterID, typename ClusterWeight, typename EdgeWeight>
struct ClusterSelectionState {
  Random &local_rand;
  NodeID u;
  typename std::type_identity<ClusterWeight>::type u_weight;
  ClusterID initial_cluster;
  ClusterWeight initial_cluster_weight;
  ClusterID best_cluster;
  EdgeWeight best_gain;
  ClusterWeight best_cluster_weight;
  EdgeWeight overall_best_gain;
  ClusterID current_cluster;
  EdgeWeight current_gain;
  ClusterWeight current_cluster_weight;
};

template <typename ClusterID, typename EdgeWeight> struct LocalClusterSelectionState {
  EdgeWeight best_gain;
  ClusterID best_cluster;
  EdgeWeight favored_cluster_gain;
  ClusterID favored_cluster;
};

template <typename NodeID> struct StatelessNeighborPolicy {
  [[nodiscard]] bool accept(const NodeID, const NodeID) const {
    return true;
  }

  [[nodiscard]] bool activate(const NodeID) const {
    return true;
  }

  [[nodiscard]] bool skip(const NodeID) const {
    return false;
  }
};

} // namespace kaminpar::lp
