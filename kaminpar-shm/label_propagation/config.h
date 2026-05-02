/*******************************************************************************
 * Configuration types for label propagation.
 *
 * @file:   config.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/random.h"

namespace kaminpar {

struct LabelPropagationConfig {
  // Data structures used to accumulate edge weights for gain value calculation
  using RatingMap = ::kaminpar::RatingMap<shm::EdgeWeight, shm::NodeID>;
  using ConcurrentRatingMap = ConcurrentFastResetArray<shm::EdgeWeight, shm::NodeID>;
  using GrowingRatingMap = DynamicRememberingFlatMap<shm::NodeID, shm::EdgeWeight>;

  // Data type for cluster IDs and weights
  using ClusterID = void;
  using ClusterWeight = void;

  // Approx. number of edges per work unit
  static constexpr shm::NodeID kMinChunkSize = 1024;

  // Nodes per permutation unit: when iterating over nodes in a chunk, we divide
  // them into permutation units, iterate over permutation orders in random
  // order, and iterate over nodes inside a permutation unit in random order.
  static constexpr shm::NodeID kPermutationSize = 64;

  // When randomizing the node order inside a permutation unit, we pick a random
  // permutation from a pool of permutations. This constant determines the pool
  // size.
  static constexpr std::size_t kNumberOfNodePermutations = 64;

  // When computing a new cluster for each node in an iteration, nodes that use more than or equal
  // to the threshold amount of entires in the rating map are processed in a second phase
  // sequentially.
  static constexpr std::size_t kRatingMapThreshold = 10000;

  // If true, we count the number of empty clusters
  static constexpr bool kTrackClusterCount = true;

  // If true, match singleton clusters in 2-hop distance
  static constexpr bool kUseTwoHopClustering = false;

  static constexpr bool kUseActualGain = false;

  static constexpr bool kUseActiveSetStrategy = true;
  static constexpr bool kUseLocalActiveSetStrategy = false;
};

template <typename ClusterID, typename ClusterWeight, typename NodeWeight, typename EdgeWeight>
struct ClusterSelectionState {
  Random &local_rand;
  ClusterID u;
  NodeWeight u_weight;
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

template <typename ClusterID, typename EdgeWeight>
struct LocalClusterSelectionState {
  EdgeWeight best_gain;
  ClusterID best_cluster;
  EdgeWeight favored_cluster_gain;
  ClusterID favored_cluster;
};

template <typename NodeID> struct AbstractChunk {
  NodeID start;
  NodeID end;
};

struct Bucket {
  std::size_t start;
  std::size_t end;
};

} // namespace kaminpar
