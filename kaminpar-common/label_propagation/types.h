/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   types.h
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>

#include "kaminpar-common/random.h"

namespace kaminpar::lp {

#if defined(__GNUC__) || defined(__clang__)
#define KAMINPAR_LP_INLINE inline __attribute__((always_inline))
#else
#define KAMINPAR_LP_INLINE inline
#endif

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

enum class CandidateComparison {
  WORSE,
  EQUIVALENT,
  BETTER,
};

template <typename NodeID, typename ClusterID> struct Initialization {
  NodeID num_nodes;
  NodeID num_active_nodes;
  ClusterID num_clusters;
};

template <typename NodeID> struct NodeLimits {
  NodeID max_degree = std::numeric_limits<NodeID>::max();
  NodeID max_neighbors = std::numeric_limits<NodeID>::max();
};

struct RatingConfig {
  RatingMapStrategy strategy = RatingMapStrategy::SINGLE_PHASE;
  std::size_t large_map_threshold = 10000;
  bool relabel_before_second_phase = false;
};

struct ActiveSetConfig {
  ActiveSetStrategy strategy = ActiveSetStrategy::NONE;
};

struct ClusterSelectionConfig {
  TieBreakingStrategy tie_breaking_strategy = TieBreakingStrategy::GEOMETRIC;
  bool use_actual_gain = false;
  bool track_favored_clusters = false;
};

template <typename ClusterID> struct StopConfig {
  ClusterID desired_clusters = 0;
  bool track_cluster_count = false;
};

template <typename NodeID, typename ClusterID> struct PassConfig {
  NodeLimits<NodeID> nodes;
  RatingConfig rating;
  ActiveSetConfig active_set;
  ClusterSelectionConfig selection;
  StopConfig<ClusterID> stopping;
};

template <typename NodeID, typename ClusterID> using Options = PassConfig<NodeID, ClusterID>;

template <
    typename NodeID,
    typename NodeWeight,
    typename ClusterID,
    typename ClusterWeight,
    typename EdgeWeight>
struct NodeContext {
  using NodeIDType = NodeID;
  using NodeWeightType = NodeWeight;
  using ClusterIDType = ClusterID;
  using ClusterWeightType = ClusterWeight;
  using EdgeWeightType = EdgeWeight;

  Random &rand;
  NodeID node;
  NodeWeight node_weight;
  ClusterID initial_cluster;
  ClusterWeight initial_cluster_weight;
  EdgeWeight gain_delta;
  bool track_favored_cluster;
};

template <typename ClusterID, typename ClusterWeight, typename EdgeWeight> struct ClusterCandidate {
  ClusterID cluster;
  EdgeWeight gain;
  ClusterWeight weight;
};

template <typename ClusterID, typename ClusterWeight, typename EdgeWeight> struct ClusterChoice {
  ClusterID best_cluster;
  EdgeWeight best_gain;
  ClusterWeight best_cluster_weight;
  ClusterID favored_cluster;
  EdgeWeight favored_gain;
};

template <typename NodeID, typename EdgeWeight> struct MoveCandidate {
  NodeID node;
  EdgeWeight gain;
};

template <typename NodeID, typename NodeWeight, typename ClusterID, typename EdgeWeight>
struct NodeMove {
  NodeID node;
  NodeWeight node_weight;
  ClusterID old_cluster;
  ClusterID new_cluster;
  EdgeWeight gain;
  bool valid = false;
};

template <typename NodeID, typename ClusterID, typename EdgeWeight> struct PassStats {
  NodeID processed_nodes = 0;
  NodeID moved_nodes = 0;
  ClusterID removed_clusters = 0;
  EdgeWeight expected_total_gain = 0;
};

template <typename NodeID, typename ClusterID, typename EdgeWeight> struct PassResult {
  NodeID processed_nodes = 0;
  NodeID moved_nodes = 0;
  ClusterID removed_clusters = 0;
  EdgeWeight expected_total_gain = 0;
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
