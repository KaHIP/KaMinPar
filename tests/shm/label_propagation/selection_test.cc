/*******************************************************************************
 * Unit tests for the cluster selection strategies.
 *
 * @file:   selection_test.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include <limits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-shm/label_propagation/config.h"
#include "kaminpar-shm/label_propagation/overload_aware_selection.h"
#include "kaminpar-shm/label_propagation/simple_gain_selection.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/random.h"

namespace kaminpar::lp::testing {

using NodeID = shm::NodeID;
using ClusterWeight = shm::NodeWeight;
using EdgeWeight = shm::EdgeWeight;
using State = ClusterSelectionState<NodeID, ClusterWeight, ClusterWeight, EdgeWeight>;

// A minimal rating map for tests: just a vector of (cluster, rating) pairs.
struct MockMap {
  std::vector<std::pair<NodeID, EdgeWeight>> _entries;
  const std::vector<std::pair<NodeID, EdgeWeight>> &entries() const {
    return _entries;
  }
};

// Minimal cluster ops for SimpleGainClusterSelection tests.
struct SimpleOps {
  std::vector<ClusterWeight> weights;
  ClusterWeight max_weight = 1000;
  bool accept = true;

  NodeID cluster(NodeID u) {
    return u;
  }

  ClusterWeight cluster_weight(NodeID c) {
    return weights[c];
  }

  ClusterWeight max_cluster_weight(NodeID) {
    return max_weight;
  }

  ClusterWeight min_cluster_weight(NodeID) {
    return 0;
  }

  bool accept_cluster(NodeID, NodeID) {
    return accept;
  }
};

// Minimal cluster ops for OverloadAwareClusterSelection tests.
struct OverloadOps {
  std::vector<ClusterWeight> weights;
  std::vector<ClusterWeight> max_weights;
  ClusterWeight min_weight = 0;

  NodeID cluster(NodeID u) {
    return u;
  }

  ClusterWeight cluster_weight(NodeID c) {
    return weights[c];
  }

  ClusterWeight max_cluster_weight(NodeID c) {
    return max_weights[c];
  }

  ClusterWeight min_cluster_weight(NodeID) {
    return min_weight;
  }

  bool accept_cluster(NodeID, NodeID) {
    return true;
  }
};

// Initialize a state with sensible defaults for testing.
// The initial best_gain is set to a very low value so any gain beats it.
State make_state(
    NodeID u,
    ClusterWeight u_weight,
    NodeID initial_cluster,
    ClusterWeight initial_weight
) {
  return State{
      Random::instance(),
      u,
      u_weight,
      initial_cluster,
      initial_weight,
      initial_cluster,
      std::numeric_limits<EdgeWeight>::min() / 2,
      initial_weight,
      std::numeric_limits<EdgeWeight>::min() / 2,
      initial_cluster,
      0,
      0,
  };
}

// =============================================================================
// SimpleGainClusterSelection
// =============================================================================

TEST(SimpleGainSelectionTest, SelectsHighestGain) {
  SimpleOps ops;
  ops.weights = {5, 5, 5};
  SimpleGainClusterSelection<SimpleOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  MockMap map;
  map._entries = {{0, 10}, {1, 20}, {2, 5}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(3, 1, 3, 5);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  EXPECT_EQ(state.best_cluster, 1); // highest rating
}

TEST(SimpleGainSelectionTest, RejectsOverweightCluster) {
  SimpleOps ops;
  ops.weights = {5, 999, 5}; // cluster 1 is heavy
  ops.max_weight = 1000;     // u_weight = 2, so 999+2 > 1000: rejected
  SimpleGainClusterSelection<SimpleOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  MockMap map;
  map._entries = {{0, 10}, {1, 20}, {2, 5}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(3, 2, 3, 5);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  EXPECT_EQ(state.best_cluster, 0); // cluster 1 rejected due to weight
}

TEST(SimpleGainSelectionTest, RespectsAcceptCluster) {
  SimpleOps ops;
  ops.weights = {5, 5, 5};
  ops.accept = false; // no cluster is accepted by the community constraint
  SimpleGainClusterSelection<SimpleOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  MockMap map;
  map._entries = {{0, 10}, {1, 20}, {2, 5}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(3, 1, 3, 5);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  // All rejected: best_cluster stays at initial
  EXPECT_EQ(state.best_cluster, 3);
}

TEST(SimpleGainSelectionTest, UniformTieBreaking) {
  SimpleOps ops;
  ops.weights = {5, 5};
  SimpleGainClusterSelection<SimpleOps> sel(ops, shm::TieBreakingStrategy::UNIFORM);

  // Equal gains for both clusters
  MockMap map;
  map._entries = {{0, 10}, {1, 10}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(2, 1, 2, 5);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  // Must pick one of {0, 1}
  EXPECT_TRUE(state.best_cluster == 0 || state.best_cluster == 1);
}

TEST(SimpleGainSelectionTest, StoresFavoredCluster) {
  SimpleOps ops;
  ops.weights = {5, 5, 5};
  SimpleGainClusterSelection<SimpleOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  // Cluster 1 has the highest overall gain
  MockMap map;
  map._entries = {{0, 5}, {1, 15}, {2, 3}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(3, 1, 3, 5);
  const NodeID favored = sel.select_best_cluster(true, 0, state, map, tb, tbf);

  EXPECT_EQ(favored, 1);
}

// =============================================================================
// OverloadAwareClusterSelection
// =============================================================================

TEST(OverloadAwareSelectionTest, StaysIfSourceBlockAtMinWeight) {
  OverloadOps ops;
  ops.weights = {5, 10, 10}; // cluster 0 (source) has weight 5
  ops.max_weights = {100, 100, 100};
  ops.min_weight = 5; // 5 - u_weight(1) = 4 < 5: cannot leave
  OverloadAwareClusterSelection<OverloadOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  MockMap map;
  map._entries = {{1, 20}, {2, 10}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(0, 1, 0, 5);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  // Cannot move: source underweight. Selection stays at initial.
  EXPECT_EQ(state.best_cluster, 0);
}

TEST(OverloadAwareSelectionTest, SelectsHighestGainWhenNoConstraints) {
  OverloadOps ops;
  ops.weights = {5, 5, 5}; // balanced, well below max
  ops.max_weights = {1000, 1000, 1000};
  ops.min_weight = 0;
  OverloadAwareClusterSelection<OverloadOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  MockMap map;
  map._entries = {{1, 15}, {2, 10}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(0, 1, 0, 5);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  EXPECT_EQ(state.best_cluster, 1);
}

TEST(OverloadAwareSelectionTest, PrefersLessOverloadedBlockWithSameGain) {
  OverloadOps ops;
  // cluster 0 (initial): weight 150, max 100 (overload 50) -- source is overloaded
  // cluster 1: weight 150, max 100 (overload 50) -- same as source
  // cluster 2: weight 120, max 100 (overload 20) -- less overloaded than source
  ops.weights = {150, 150, 120};
  ops.max_weights = {100, 100, 100};
  ops.min_weight = 0;
  OverloadAwareClusterSelection<OverloadOps> sel(ops, shm::TieBreakingStrategy::UNIFORM);

  // Equal gain for clusters 1 and 2. The source block is overloaded, so moves are allowed.
  MockMap map;
  map._entries = {{1, 10}, {2, 10}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(0, 1, 0, 150);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  // Cluster 2 (overload 20) should beat cluster 1 (overload 50).
  EXPECT_EQ(state.best_cluster, 2);
}

TEST(OverloadAwareSelectionTest, AllowsMoveToOverloadedBlockIfReducesImbalance) {
  OverloadOps ops;
  // initial cluster is overloaded: weight 150, max 100 (overload 50)
  // target cluster: weight 110, max 100 (overload 10) -- less overloaded than source
  ops.weights = {150, 110};
  ops.max_weights = {100, 100};
  ops.min_weight = 0;
  OverloadAwareClusterSelection<OverloadOps> sel(ops, shm::TieBreakingStrategy::GEOMETRIC);

  MockMap map;
  map._entries = {{1, 10}};

  std::vector<NodeID> tb, tbf;
  auto state = make_state(0, 1, 0, 150);
  sel.select_best_cluster(false, 0, state, map, tb, tbf);

  // Moving to cluster 1 (overload 10) reduces overload vs source (overload 50)
  EXPECT_EQ(state.best_cluster, 1);
}

} // namespace kaminpar::lp::testing
