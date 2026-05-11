/*******************************************************************************
 * Composable label propagation building blocks.
 *
 * @file:   types.h
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#include "kaminpar-common/inline.h"
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

struct ExecutionConfig {
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

template <typename NodeID, typename ClusterID, typename EdgeWeight> struct alignas(64) PassStats {
  NodeID processed_nodes = 0;
  NodeID moved_nodes = 0;
  ClusterID removed_clusters = 0;
  EdgeWeight expected_total_gain = 0;

  KAMINPAR_INLINE PassStats &operator+=(const PassStats &other) {
    processed_nodes += other.processed_nodes;
    moved_nodes += other.moved_nodes;
    removed_clusters += other.removed_clusters;
    expected_total_gain += other.expected_total_gain;
    return *this;
  }
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
  static constexpr bool kAcceptsAllNeighbors = true;
  static constexpr bool kActivatesAllNeighbors = true;
  static constexpr bool kSkipsNoNodes = true;

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

template <typename NeighborPolicy, typename = void> struct AcceptsAllNeighbors : std::false_type {};

template <typename NeighborPolicy>
struct AcceptsAllNeighbors<
    NeighborPolicy,
    std::void_t<decltype(NeighborPolicy::kAcceptsAllNeighbors)>>
    : std::bool_constant<NeighborPolicy::kAcceptsAllNeighbors> {};

template <typename NeighborPolicy, typename = void>
struct ActivatesAllNeighbors : std::false_type {};

template <typename NeighborPolicy>
struct ActivatesAllNeighbors<
    NeighborPolicy,
    std::void_t<decltype(NeighborPolicy::kActivatesAllNeighbors)>>
    : std::bool_constant<NeighborPolicy::kActivatesAllNeighbors> {};

template <typename NeighborPolicy, typename = void> struct SkipsNoNodes : std::false_type {};

template <typename NeighborPolicy>
struct SkipsNoNodes<NeighborPolicy, std::void_t<decltype(NeighborPolicy::kSkipsNoNodes)>>
    : std::bool_constant<NeighborPolicy::kSkipsNoNodes> {};

template <typename ClusterSelector, typename = void>
struct HasStaticUseActualGain : std::false_type {};

template <typename ClusterSelector>
struct HasStaticUseActualGain<
    ClusterSelector,
    std::void_t<decltype(ClusterSelector::kUseActualGain)>> : std::true_type {};

template <typename ClusterSelector, typename = void> struct UsesActualGain : std::false_type {};

template <typename ClusterSelector>
struct UsesActualGain<ClusterSelector, std::void_t<decltype(ClusterSelector::kUseActualGain)>>
    : std::bool_constant<ClusterSelector::kUseActualGain> {};

template <typename ClusterSelector, typename = void>
struct HasStaticTrackFavoredClusters : std::false_type {};

template <typename ClusterSelector>
struct HasStaticTrackFavoredClusters<
    ClusterSelector,
    std::void_t<decltype(ClusterSelector::kTrackFavoredClusters)>> : std::true_type {};

template <typename ClusterSelector, typename = void>
struct TracksFavoredClusters : std::false_type {};

template <typename ClusterSelector>
struct TracksFavoredClusters<
    ClusterSelector,
    std::void_t<decltype(ClusterSelector::kTrackFavoredClusters)>>
    : std::bool_constant<ClusterSelector::kTrackFavoredClusters> {};

} // namespace kaminpar::lp
