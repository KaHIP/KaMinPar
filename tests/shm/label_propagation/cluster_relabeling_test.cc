/*******************************************************************************
 * Unit tests for the cluster relabeling utility.
 *
 * @file:   cluster_relabeling_test.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-shm/label_propagation/cluster_relabeling.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::lp::testing {

using NodeID = shm::NodeID;

struct MockClusterOps {
  std::vector<NodeID> clusters;
  bool reassign_called = false;

  NodeID cluster(NodeID u) {
    return clusters[u];
  }

  void move_node(NodeID u, NodeID c) {
    clusters[u] = c;
  }

  void reassign_cluster_weights(const StaticArray<NodeID> &, NodeID) {
    reassign_called = true;
  }
};

TEST(ClusterRelabelingTest, RemapsToConsecutiveIDs) {
  // clusters [0, 0, 2, 2, 4, 4] uses IDs {0, 2, 4} -> should map to {0, 1, 2}
  MockClusterOps ops;
  ops.clusters = {0, 0, 2, 2, 4, 4};

  relabel_clusters<NodeID, NodeID>(6, ops, 3, nullptr, nullptr);

  EXPECT_EQ(ops.clusters[0], 0);
  EXPECT_EQ(ops.clusters[1], 0);
  EXPECT_EQ(ops.clusters[2], 1);
  EXPECT_EQ(ops.clusters[3], 1);
  EXPECT_EQ(ops.clusters[4], 2);
  EXPECT_EQ(ops.clusters[5], 2);
}

TEST(ClusterRelabelingTest, SingletonClustersRemappedCorrectly) {
  // each node is its own cluster: [0,1,2,3] -> should stay [0,1,2,3]
  MockClusterOps ops;
  ops.clusters = {0, 1, 2, 3};

  relabel_clusters<NodeID, NodeID>(4, ops, 4, nullptr, nullptr);

  EXPECT_EQ(ops.clusters[0], 0);
  EXPECT_EQ(ops.clusters[1], 1);
  EXPECT_EQ(ops.clusters[2], 2);
  EXPECT_EQ(ops.clusters[3], 3);
}

TEST(ClusterRelabelingTest, SetsMoved) {
  // nodes not in their own cluster get moved[u] = 1
  MockClusterOps ops;
  ops.clusters = {0, 0, 2, 2, 4, 4};

  StaticArray<std::uint8_t> moved(6);

  relabel_clusters<NodeID, NodeID>(6, ops, 3, nullptr, &moved);

  EXPECT_EQ(moved[0], 0); // node 0 is in cluster 0 (self)
  EXPECT_EQ(moved[1], 1); // node 1 is in cluster 0 (not self)
  EXPECT_EQ(moved[2], 0); // node 2 is in cluster 2 (self)
  EXPECT_EQ(moved[3], 1); // node 3 is in cluster 2 (not self)
  EXPECT_EQ(moved[4], 0); // node 4 is in cluster 4 (self)
  EXPECT_EQ(moved[5], 1); // node 5 is in cluster 4 (not self)
}

TEST(ClusterRelabelingTest, RelabelsFavoredClusters) {
  // favored_clusters use old cluster IDs and must be updated too
  MockClusterOps ops;
  ops.clusters = {0, 0, 2, 2};

  // node i favors the "other" cluster
  StaticArray<NodeID> favored = static_array::create<NodeID>({2, 2, 0, 0});

  relabel_clusters<NodeID, NodeID>(4, ops, 2, &favored, nullptr);

  // old cluster 0 -> new cluster 0, old cluster 2 -> new cluster 1
  EXPECT_EQ(favored[0], 1); // was 2, now 1
  EXPECT_EQ(favored[1], 1);
  EXPECT_EQ(favored[2], 0); // was 0, stays 0
  EXPECT_EQ(favored[3], 0);
}

TEST(ClusterRelabelingTest, CallsReassignClusterWeights) {
  MockClusterOps ops;
  ops.clusters = {0, 0, 2, 2};

  relabel_clusters<NodeID, NodeID>(4, ops, 2, nullptr, nullptr);

  EXPECT_TRUE(ops.reassign_called);
}

} // namespace kaminpar::lp::testing
