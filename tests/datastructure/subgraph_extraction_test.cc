#include "algorithm/graph_utils.h"
#include "matcher.h"
#include "tests.h"

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Lt;
using ::testing::Ne;
using ::testing::UnorderedElementsAre;
using namespace ::kaminpar::test;

namespace kaminpar {
TEST(SubgraphExtractionTest, ExtractsIsolatedNodes) {
  Graph graph{create_graph({0, 0, 0, 0, 0}, {})};
  PartitionedGraph p_graph{create_p_graph(graph, 4, {0, 1, 2, 3})};

  SubgraphMemory memory{p_graph};
  auto result = extract_subgraphs(p_graph, memory);

  EXPECT_THAT(result.subgraphs[0].n(), 1);
  EXPECT_THAT(result.subgraphs[1].n(), 1);
  EXPECT_THAT(result.subgraphs[2].n(), 1);
  EXPECT_THAT(result.subgraphs[3].n(), 1);
  EXPECT_THAT(result.subgraphs[0].m(), 0);
  EXPECT_THAT(result.subgraphs[1].m(), 0);
  EXPECT_THAT(result.subgraphs[2].m(), 0);
  EXPECT_THAT(result.subgraphs[3].m(), 0);
}

TEST(SubgraphExtractionTest, ExtractsEdges) {
  Graph graph{create_graph({0, 1, 2, 3, 4}, {1, 0, 3, 2})};
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 0, 1, 1})};

  SubgraphMemory memory{p_graph};
  auto result = extract_subgraphs(p_graph, memory);

  EXPECT_THAT(result.subgraphs[0].n(), 2);
  EXPECT_THAT(result.subgraphs[1].n(), 2);
  EXPECT_THAT(result.subgraphs[0].m(), 2);
  EXPECT_THAT(result.subgraphs[1].m(), 2);
  EXPECT_THAT(result.subgraphs[0].edge_target(0), AnyOf(0, 1));
  EXPECT_THAT(result.subgraphs[0].edge_target(1), AnyOf(1, 0));
  EXPECT_THAT(result.subgraphs[0].edge_target(0), Ne(result.subgraphs[0].edge_target(1)));
  EXPECT_THAT(result.subgraphs[1].edge_target(0), AnyOf(0, 1));
  EXPECT_THAT(result.subgraphs[1].edge_target(1), AnyOf(1, 0));
  EXPECT_THAT(result.subgraphs[1].edge_target(0), Ne(result.subgraphs[0].edge_target(1)));
}


TEST(SubgraphExtractionTest, ExtractsPathCutInTwo) {
  Graph graph{create_graph({0, 1, 3, 5, 6}, {1, 0, 2, 1, 3, 2})};
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 0, 1, 1})};

  SubgraphMemory memory{p_graph};
  auto result = extract_subgraphs(p_graph, memory);

  EXPECT_THAT(result.subgraphs[0].n(), 2);
  EXPECT_THAT(result.subgraphs[1].n(), 2);
  EXPECT_THAT(result.subgraphs[0].m(), 2);
  EXPECT_THAT(result.subgraphs[1].m(), 2);
  EXPECT_THAT(result.subgraphs[0].edge_target(0), AnyOf(0, 1));
  EXPECT_THAT(result.subgraphs[0].edge_target(1), AnyOf(1, 0));
  EXPECT_THAT(result.subgraphs[0].edge_target(0), Ne(result.subgraphs[0].edge_target(1)));
  EXPECT_THAT(result.subgraphs[1].edge_target(0), AnyOf(0, 1));
  EXPECT_THAT(result.subgraphs[1].edge_target(1), AnyOf(1, 0));
  EXPECT_THAT(result.subgraphs[1].edge_target(0), Ne(result.subgraphs[0].edge_target(1)));
}

TEST(SubgraphExtractionTest, ComplexTrianglesWeightedExampleWorks) {
  // 0 -- 1 -- 2 -- 3
  //  \  /      \  /
  //   4 -------- 5
  //    \        /
  //     6 ---- 7
  //      \    /
  //         8
  // weights shifted by one
  // edges weight = sum of incident node weights
  // each triangle in one block
  Graph graph{create_graph({0, 2, 5, 8, 10, 14, 18, 21, 24, 26},
                           {1, 4,
                            0, 2, 4,
                            1, 3, 5,
                            2, 5,
                            0, 1, 5, 6,
                            2, 3, 4, 7,
                            4, 7, 8,
                            5, 6, 8,
                            6, 7},
                           {1, 2, 3, 4, 5, 6, 7, 8, 9},
                           {3, 6,
                            3, 5, 7,
                            5, 7, 9,
                            7, 10,
                            6, 7, 11, 12,
                            9, 10, 11, 14,
                            12, 15, 16,
                            14, 15, 17,
                            16, 17})};
  PartitionedGraph p_graph{create_p_graph(graph, 3, {0, 0, 1, 1, 0, 1, 2, 2, 2}, {4, 5, 6})};

  SubgraphMemory memory{p_graph.n(), 15, p_graph.m(), p_graph.graph().is_node_weighted(),
                        p_graph.graph().is_edge_weighted()};
  auto result = extract_subgraphs(p_graph, memory);

  EXPECT_THAT(result.subgraphs[0].n(), 3);
  EXPECT_THAT(result.subgraphs[1].n(), 3);
  EXPECT_THAT(result.subgraphs[2].n(), 3);
  EXPECT_THAT(result.subgraphs[0].m(), 6);
  EXPECT_THAT(result.subgraphs[1].m(), 6);
  EXPECT_THAT(result.subgraphs[2].m(), 6);

  EXPECT_THAT(result.subgraphs[0], HasEdgeWithWeightedEndpoints(1, 2));
  EXPECT_THAT(result.subgraphs[0], HasEdgeWithWeightedEndpoints(1, 5));
  EXPECT_THAT(result.subgraphs[0], HasEdgeWithWeightedEndpoints(2, 5));
  EXPECT_THAT(result.subgraphs[0].total_edge_weight(), 32);

  EXPECT_THAT(result.subgraphs[1], HasEdgeWithWeightedEndpoints(3, 4));
  EXPECT_THAT(result.subgraphs[1], HasEdgeWithWeightedEndpoints(3, 6));
  EXPECT_THAT(result.subgraphs[1], HasEdgeWithWeightedEndpoints(4, 6));

  EXPECT_THAT(result.subgraphs[2], HasEdgeWithWeightedEndpoints(7, 8));
  EXPECT_THAT(result.subgraphs[2], HasEdgeWithWeightedEndpoints(7, 9));
  EXPECT_THAT(result.subgraphs[2], HasEdgeWithWeightedEndpoints(8, 9));
}
}