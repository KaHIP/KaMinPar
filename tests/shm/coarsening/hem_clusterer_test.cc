#include <algorithm>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/coarsening/clustering/hem_clusterer.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

using ::testing::Eq;

namespace kaminpar::shm::testing {

namespace {

StaticArray<NodeID> compute_hem_clustering(Graph &graph, const NodeWeight max_cluster_weight) {
  Context ctx = create_default_context();
  HEMClustering hem(ctx.coarsening);
  hem.set_max_cluster_weight(max_cluster_weight);

  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  hem.compute_clustering(clustering, graph, true);
  return clustering;
}

std::unordered_map<NodeID, std::vector<NodeID>>
collect_clusters(const StaticArray<NodeID> &clustering) {
  std::unordered_map<NodeID, std::vector<NodeID>> clusters;
  for (NodeID u = 0; u < clustering.size(); ++u) {
    clusters[clustering[u]].push_back(u);
  }
  return clusters;
}

} // namespace

TEST(HEMClustererTest, MatchesSingleHeavyEdge) {
  Graph graph = make_graph({0, 1, 2}, {1, 0});
  StaticArray<NodeID> clustering = compute_hem_clustering(graph, 2);

  EXPECT_THAT(clustering[0], Eq(0));
  EXPECT_THAT(clustering[1], Eq(0));
}

TEST(HEMClustererTest, TwoHopMatchesNodesWithCommonHeavyNeighbor) {
  GraphBuilder builder;
  builder.new_node(10);
  builder.new_edge(1, 4);
  builder.new_edge(2, 4);
  builder.new_edge(3, 4);
  builder.new_edge(4, 4);
  for (NodeID leaf = 1; leaf <= 4; ++leaf) {
    builder.new_node(1);
    builder.new_edge(0, 4);
  }
  Graph graph = builder.build();

  StaticArray<NodeID> clustering = compute_hem_clustering(graph, 2);
  const auto clusters = collect_clusters(clustering);

  EXPECT_THAT(clustering[0], Eq(0));
  EXPECT_THAT(clusters.size(), Eq(3));

  int leaf_pairs = 0;
  for (const auto &[cluster, nodes] : clusters) {
    if (cluster == 0) {
      continue;
    }

    EXPECT_THAT(nodes.size(), Eq(2));
    EXPECT_THAT(std::ranges::all_of(nodes, [](const NodeID u) { return u != 0; }), Eq(true));
    ++leaf_pairs;
  }
  EXPECT_THAT(leaf_pairs, Eq(2));
}

TEST(HEMClustererTest, RespectsCommunities) {
  Graph graph = make_graph({0, 1, 2}, {1, 0});
  Context ctx = create_default_context();
  HEMClustering hem(ctx.coarsening);
  hem.set_max_cluster_weight(2);

  const std::vector<NodeID> communities = {0, 1};
  hem.set_communities(communities);

  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  hem.compute_clustering(clustering, graph, true);

  EXPECT_THAT(clustering[0], Eq(0));
  EXPECT_THAT(clustering[1], Eq(1));
}

TEST(HEMClustererTest, RespectsDesiredClusterCount) {
  Graph graph = make_matching_graph(2);
  Context ctx = create_default_context();
  HEMClustering hem(ctx.coarsening);
  hem.set_max_cluster_weight(2);
  hem.set_desired_cluster_count(graph.n());

  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  hem.compute_clustering(clustering, graph, true);

  for (const NodeID u : graph.csr_graph().nodes()) {
    EXPECT_THAT(clustering[u], Eq(u));
  }
}

} // namespace kaminpar::shm::testing
