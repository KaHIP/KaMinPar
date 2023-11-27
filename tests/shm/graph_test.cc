#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"
#include "tests/shm/matchers.h"
#include "tests/shm/test_helpers.h"

#include "kaminpar-shm/graphutils/subgraph_extractor.h"

namespace kaminpar::shm::testing {
class AWeightedGridGraph : public ::testing::Test {
public:
  // 0|1--- 1|2--- 2|4--- 3|8
  // |    / |    / |    / |
  // 4|16---5|32---6|64---7|128
  AWeightedGridGraph()
      : graph{create_graph(
            {0, 2, 6, 10, 13, 16, 20, 24, 26},
            {1, 4, 0, 4, 5, 2, 1, 5, 6, 3, 2, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3},
            {1, 2, 4, 8, 16, 32, 64, 128},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
        )} {}

  Graph graph;
};

//
// Extracting block induced subgraphs from a partitioned graph
//

TEST_F(AWeightedGridGraph, ExtractingBlockInducedSubgraphsWorkHorizontally) {
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 0, 0, 0, 1, 1, 1, 1})};
  graph::SubgraphMemory memory{p_graph};
  const auto [subgraphs, node_mapping, positions] = extract_subgraphs(p_graph, 2, memory);
  const auto &s_graph0 = subgraphs[0];
  const auto &s_graph1 = subgraphs[1];

  EXPECT_EQ(s_graph0.n(), 4);
  EXPECT_EQ(s_graph0.m(), 6);
  EXPECT_EQ(s_graph1.n(), 4);
  EXPECT_EQ(s_graph1.m(), 6);

  EXPECT_THAT(s_graph0, HasEdgeWithWeightedEndpoints(1, 2));
  EXPECT_THAT(s_graph0, HasEdgeWithWeightedEndpoints(2, 4));
  EXPECT_THAT(s_graph0, HasEdgeWithWeightedEndpoints(4, 8));
  EXPECT_THAT(s_graph1, HasEdgeWithWeightedEndpoints(16, 32));
  EXPECT_THAT(s_graph1, HasEdgeWithWeightedEndpoints(32, 64));
  EXPECT_THAT(s_graph1, HasEdgeWithWeightedEndpoints(64, 128));
}

TEST_F(AWeightedGridGraph, ExtractingEmptyBlockInducedSubgraphWorks) {
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 0, 0, 0, 0, 0, 0, 0})};
  graph::SubgraphMemory memory{p_graph};
  const auto [subgraphs, node_mapping, positions] = extract_subgraphs(p_graph, 2, memory);
  const auto &s_graph0 = subgraphs[0];
  const auto &s_graph1 = subgraphs[1];

  EXPECT_EQ(s_graph0.n(), graph.n());
  EXPECT_EQ(s_graph0.m(), graph.m());
  EXPECT_EQ(s_graph1.n(), 0);
  EXPECT_EQ(s_graph1.m(), 0);
}

//
// Node and edge weights
//

TEST_F(AWeightedGridGraph, InitialNodeWeightingWorks) {
  for (const NodeID u : graph.nodes()) {
    EXPECT_EQ(graph.node_weight(u), 1 << u);
  }
}

TEST_F(AWeightedGridGraph, InitialEdgeWeightingWorks) {
  for (const EdgeID e : graph.edges()) {
    EXPECT_EQ(graph.edge_weight(e), 1);
  }
}

TEST_F(AWeightedGridGraph, InitialTotalNodeWeightWorks) {
  EXPECT_EQ(graph.total_node_weight(),
            (1 << graph.n()) - 1); // graph has node weights 1, 2, 4, ...
}

TEST_F(AWeightedGridGraph, InitialTotalEdgeWeightWorks) {
  EXPECT_EQ(graph.total_edge_weight(),
            graph.m()); // graph has edge weights 1, 1, 1, ...
}

//
// Degree of nodes
//

TEST_F(AWeightedGridGraph, DegreeWorks) {
  EXPECT_EQ(graph.degree(0), 2);
  EXPECT_EQ(graph.degree(1), 4);
  EXPECT_EQ(graph.degree(6), 4);
  EXPECT_EQ(graph.degree(7), 2);
}

TEST(GraphTest, DegreeWorksForLeaves) {
  const Graph graph{create_graph({0, 1, 2}, {1, 0})};
  EXPECT_EQ(graph.degree(0), 1);
  EXPECT_EQ(graph.degree(1), 1);
}

TEST(GraphTest, DegreeWorksForGraphWithIsolatedNodes) {
  const Graph graph{create_graph({0, 1, 1, 1, 2}, {3, 0})};
  EXPECT_EQ(graph.degree(0), 1);
  EXPECT_EQ(graph.degree(1), 0);
  EXPECT_EQ(graph.degree(2), 0);
  EXPECT_EQ(graph.degree(3), 1);
}

//
// Block weights in partitioned graph
//

TEST_F(AWeightedGridGraph, InitialBlockWeightsAreCorrect) {
  PartitionedGraph p_graph{create_p_graph(graph, 4, {0, 0, 1, 1, 2, 2, 3, 3})};
  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 12);
  EXPECT_EQ(p_graph.block_weight(2), 48);
  EXPECT_EQ(p_graph.block_weight(3), 192);
}

TEST_F(AWeightedGridGraph, BlockWeightsAreUpdatedOnNodeMove) {
  PartitionedGraph p_graph{create_p_graph(graph, 4, {0, 0, 1, 1, 2, 2, 3, 3})};
  p_graph.set_block(0, 1);
  EXPECT_EQ(p_graph.block_weight(0), 2);
  EXPECT_EQ(p_graph.block_weight(1), 13);
}

//
// Blocks
//

TEST(GraphTest, PartitionedGraphReturnsCorrectNumberOfBlocks) {
  Graph graph{create_graph({0}, {})};
  const PartitionedGraph p_graph{create_p_graph(&graph, 4)};
  EXPECT_EQ(p_graph.k(), 4);
}

TEST(GraphTest, InitialBlocksAreCorrect) {
  Graph graph{create_graph({0, 0, 0, 0, 0}, {})};
  const PartitionedGraph p_graph{create_p_graph(&graph, 4, {0, 1, 2, 3})};
  for (const NodeID u : {0, 1, 2, 3}) {
    EXPECT_EQ(p_graph.block(u), u);
  }
}

TEST(GraphTest, ChangingBlocksWorks) {
  Graph graph{create_graph({0, 0, 0, 0, 0}, {})};
  PartitionedGraph p_graph{create_p_graph(&graph, 4, {0, 1, 2, 3})};
  p_graph.set_block(0, 1);
  EXPECT_EQ(p_graph.block(0), 1);
}

//
// Degree buckets
//

TEST(GraphTest, IfBucketsAreDisabledNodesAreInFirstBucket) {
  Graph graph{graphs::grid(4, 4)};
  EXPECT_EQ(16, graph.bucket_size(0));
  for (std::size_t bucket = 1; bucket < graph.number_of_buckets(); ++bucket) {
    EXPECT_EQ(0, graph.bucket_size(bucket));
  }
}

TEST(GraphTest, PutsIsolatedNodesInCorrectBucket) {
  Graph graph{graphs::empty(10, true)};
  EXPECT_EQ(10, graph.bucket_size(0));
  for (std::size_t bucket = 1; bucket < graph.number_of_buckets(); ++bucket) {
    EXPECT_EQ(0, graph.bucket_size(bucket));
  }
  EXPECT_EQ(0, graph.first_node_in_bucket(0));
  EXPECT_EQ(10, graph.first_node_in_bucket(1));
}

TEST(GraphTest, PutsMatchingInCorrectBucket) {
  Graph graph{graphs::matching(10, true)};
  EXPECT_EQ(0, graph.bucket_size(0));
  EXPECT_EQ(20, graph.bucket_size(1));
  for (std::size_t bucket = 2; bucket < graph.number_of_buckets(); ++bucket) {
    EXPECT_EQ(0, graph.bucket_size(bucket));
  }
  EXPECT_EQ(0, graph.first_node_in_bucket(0));
  EXPECT_EQ(0, graph.first_invalid_node_in_bucket(0));
  EXPECT_EQ(0, graph.first_node_in_bucket(1));
  EXPECT_EQ(20, graph.first_invalid_node_in_bucket(1));
}

TEST(GraphTest, PutsAxeInCorrectBuckets) {
  /*
   *      x
   *     / \
   * x--x---x  x
   *     \ /
   *      x
   */
  Graph graph{create_graph({0, 0, 1, 3, 5, 8, 12}, {5, 4, 5, 4, 5, 2, 3, 5, 1, 2, 3, 4}, true)};
  EXPECT_EQ(1, graph.bucket_size(0)); // deg 0
  EXPECT_EQ(1, graph.bucket_size(1)); // deg 1
  EXPECT_EQ(3, graph.bucket_size(2)); // deg 2, 3
  EXPECT_EQ(1, graph.bucket_size(3)); // deg 4, 5, 6, 7
  EXPECT_EQ(0, graph.first_node_in_bucket(0));
  EXPECT_EQ(1, graph.first_invalid_node_in_bucket(0));
  EXPECT_EQ(1, graph.first_node_in_bucket(1));
  EXPECT_EQ(2, graph.first_invalid_node_in_bucket(1));
  EXPECT_EQ(2, graph.first_node_in_bucket(2));
  EXPECT_EQ(5, graph.first_invalid_node_in_bucket(2));
  EXPECT_EQ(5, graph.first_node_in_bucket(3));
  EXPECT_EQ(6, graph.first_invalid_node_in_bucket(3));
}

TEST(GraphTest, LowestDegreeInBucketWorks) {
  EXPECT_EQ(lowest_degree_in_bucket<NodeID>(0), 0);
  EXPECT_EQ(lowest_degree_in_bucket<NodeID>(1), 1);
  EXPECT_EQ(lowest_degree_in_bucket<NodeID>(2), 2);
  EXPECT_EQ(lowest_degree_in_bucket<NodeID>(3), 4);
}
} // namespace kaminpar::shm::testing
