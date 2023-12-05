#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"
#include "tests/shm/matchers.h"
#include "tests/shm/test_helpers.h"

#include "kaminpar-shm/graphutils/cluster_contraction.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Lt;
using ::testing::UnorderedElementsAre;

namespace kaminpar::shm::testing {
TEST(ParallelContractionTest, ContractingToSingleNodeWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};

  for (const NodeID cluster : {0, 1, 2, 3}) {
    auto [c_graph, c_mapping, m_ctx] =
        graph::contract(graph, scalable_vector<NodeID>{cluster, cluster, cluster, cluster});
    EXPECT_THAT(c_graph.n(), 1);
    EXPECT_THAT(c_graph.m(), 0);
    EXPECT_THAT(c_graph.node_weight(0), graph.total_node_weight());
  }
}

TEST(ParallelContractionTest, ContractingToSingletonsWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 2);
  graph = change_node_weight(std::move(graph), 2, 3);
  graph = change_node_weight(std::move(graph), 3, 4);

  auto [c_graph, c_mapping, m_ctx] = graph::contract(graph, scalable_vector<NodeID>{0, 1, 2, 3});
  EXPECT_THAT(c_graph.n(), graph.n());
  EXPECT_THAT(c_graph.m(), graph.m());
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), graph.total_edge_weight());

  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 2));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 3));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(2, 4));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(3, 4));
}

TEST(ParallelContractionTest, ContractingAllNodesButOneWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};

  // 0--1
  // |  |
  // 2--3
  auto [c_graph, c_mapping, m_ctx] = graph::contract(graph, scalable_vector<NodeID>{0, 1, 1, 1});
  EXPECT_THAT(c_graph.n(), 2);
  EXPECT_THAT(c_graph.m(), 2); // one undirected edge
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), 2 * 2);
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 3));
}

TEST(ParallelContractionTest, ContractingGridHorizontallyWorks) {
  Graph graph{graphs::grid(2, 4)}; // two rows, 4 columns, organized row by row
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 2);
  graph = change_node_weight(std::move(graph), 2, 3);
  graph = change_node_weight(std::move(graph), 3, 4);
  graph = change_node_weight(std::move(graph), 4, 10);
  graph = change_node_weight(std::move(graph), 5, 20);
  graph = change_node_weight(std::move(graph), 6, 30);
  graph = change_node_weight(std::move(graph), 7, 40);

  auto [c_graph, c_mapping, m_ctx] =
      graph::contract(graph, scalable_vector<NodeID>{0, 1, 2, 3, 0, 1, 2, 3});
  EXPECT_THAT(c_graph.n(), 4);
  EXPECT_THAT(c_graph.m(), 2 * 3);
  EXPECT_THAT(c_graph.raw_node_weights(), UnorderedElementsAre(11, 22, 33, 44));
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), 4 * 3);
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(11, 22));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(22, 33));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(33, 44));
}

TEST(ParallelContractionTest, ContractingGridVerticallyWorks) {
  Graph graph{graphs::grid(4, 2)}; // four columns, two rows, organized row by row
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 10);
  graph = change_node_weight(std::move(graph), 2, 2);
  graph = change_node_weight(std::move(graph), 3, 20);
  graph = change_node_weight(std::move(graph), 4, 3);
  graph = change_node_weight(std::move(graph), 5, 30);
  graph = change_node_weight(std::move(graph), 6, 4);
  graph = change_node_weight(std::move(graph), 7, 40);

  auto [c_graph, c_mapping, m_ctx] =
      graph::contract(graph, scalable_vector<NodeID>{0, 0, 2, 2, 4, 4, 6, 6});
  EXPECT_THAT(c_graph.n(), 4);
  EXPECT_THAT(c_graph.m(), 2 * 3);
  EXPECT_THAT(c_graph.raw_node_weights(), UnorderedElementsAre(11, 22, 33, 44));
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), 4 * 3);
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(11, 22));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(22, 33));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(33, 44));
}

//
// Pseudo-sorting the nodes of a graph by degree
//

TEST(GraphPermutationTest, PermutationByNodeDegreeIsCorrect) {
  // 5 3
  //   |
  // 1-2-0
  //   |/
  //   4
  const StaticArray<EdgeID> nodes = create_static_array<EdgeID>({0, 2, 3, 7, 8, 10, 10});

  const auto permutations = graph::sort_by_degree_buckets(nodes);
  const auto &permutation = permutations.old_to_new;
  EXPECT_THAT(permutation[0], AllOf(Ge(2), Le(3)));
  EXPECT_THAT(permutation[1], AllOf(Ge(0), Le(1)));
  EXPECT_EQ(permutation[2], 4);
  EXPECT_THAT(permutation[3], AllOf(Ge(0), Le(1)));
  EXPECT_THAT(permutation[4], AllOf(Ge(2), Le(3)));
  EXPECT_EQ(permutation[5], 5);
}

TEST(GraphPermutationTest, MovingIsolatedNodesToBackWorks) {
  // node 0 1 2 3 4 5 6 7 8 9 10
  // deg  0 0 1 1 1 0 0 1 1 0 0
  const StaticArray<EdgeID> nodes =
      create_static_array<EdgeID>({0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5});
  const auto permutations = graph::sort_by_degree_buckets(nodes);
  const auto &permutation = permutations.old_to_new;

  EXPECT_GE(permutation[0], 5);
  EXPECT_GE(permutation[1], 5);
  EXPECT_LE(permutation[2], 4);
  EXPECT_LE(permutation[3], 4);
  EXPECT_LE(permutation[4], 4);
  EXPECT_GE(permutation[5], 5);
  EXPECT_GE(permutation[6], 5);
  EXPECT_LE(permutation[7], 4);
  EXPECT_LE(permutation[8], 4);
  EXPECT_GE(permutation[9], 5);
  EXPECT_GE(permutation[10], 5);
}

//
// Preprocessing
//

TEST(
    PreprocessingTest, PreprocessingFacadeRemovesIsolatedNodesAndAdaptsEpsilonFromUnweightedGraph
) {
  /* 0
   * 1--2--3        *--*--*
   * 4  5  6    --> *
   * |              |
   * 7--8  9        *--*--*
   * 10    11
   */
  auto nodes = create_static_array<EdgeID>({0, 0, 1, 3, 4, 5, 5, 5, 7, 8, 8, 8, 8});
  auto edges = create_static_array<NodeID>({2, 1, 3, 2, 7, 4, 8, 7});
  auto node_weights = create_static_array<NodeWeight>({});
  auto edge_weights = create_static_array<EdgeWeight>({});

  PartitionContext p_ctx;
  p_ctx.k = 2;
  p_ctx.epsilon = 0.17; // max block weight 7

  graph::rearrange_graph(p_ctx, nodes, edges, node_weights, edge_weights);

  EXPECT_EQ(nodes.size(), 7);
  EXPECT_EQ(edges.size(), 8);
  for (const NodeID v : edges) {
    EXPECT_LT(v, 7);
  } // edges are valid

  // total weight of new graph: 6, perfectly balanced block weight: 3
  // hence eps' should be 1.3333....
  EXPECT_THAT(p_ctx.epsilon, AllOf(Gt(1.33), Lt(1.34)));
}

//
// Sequential graph extraction
//
TEST(SequentialGraphExtraction, SimpleSequentialBipartitionExtractionWorks) {
  // 0--1--2     block 0
  //-|--|--
  // 3--4--5     block 1
  Graph graph{create_graph({0, 2, 5, 6, 8, 11, 12}, {1, 3, 0, 4, 2, 1, 0, 4, 3, 1, 5, 4})};
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 0, 0, 1, 1, 1})};

  graph::SubgraphMemory memory{p_graph};
  graph::SubgraphMemoryStartPosition position{0, 0};
  graph::TemporarySubgraphMemory buffer{};
  const auto [subgraphs, positions] =
      graph::extract_subgraphs_sequential(p_graph, {1, 1}, position, memory, buffer);

  for (const auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph.n(), 3);
    EXPECT_EQ(subgraph.m(), 4);
    EXPECT_THAT(degrees(subgraph), UnorderedElementsAre(1, 1, 2));
  }

  EXPECT_EQ(positions[0].nodes_start_pos, 0);
  EXPECT_EQ(positions[0].edges_start_pos, 0);
  EXPECT_EQ(positions[1].nodes_start_pos, 4);
  EXPECT_EQ(positions[1].edges_start_pos, 4);
}
} // namespace kaminpar::shm::testing
