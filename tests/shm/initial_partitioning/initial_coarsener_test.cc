/*******************************************************************************
 * @file:   initial_coarsener_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Unit tests for sequential graph contraction.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests.h"
#include "tests/shm/matcher.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/initial_partitioning/initial_coarsener.h"

using ::testing::UnorderedElementsAre;

using namespace ::kaminpar::test;

namespace kaminpar::ip {
TEST(InitialCoarsenerTest, ContractingToSingleNodeWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};

  InitialCoarsener coarsener(&graph, create_default_context().coarsening);
  for (const NodeID cluster : {0, 1, 2, 3}) {
    coarsener.TEST_mock_clustering({cluster, cluster, cluster, cluster});
    auto [c_graph, c_mapping] = coarsener.TEST_contract_clustering();
    EXPECT_THAT(c_graph.n(), 1);
    EXPECT_THAT(c_graph.m(), 0);
    EXPECT_THAT(c_graph.node_weight(0), graph.total_node_weight());
  }
}

TEST(InitialCoarsenerTest, ContractingToSingletonsWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 2);
  graph = change_node_weight(std::move(graph), 2, 3);
  graph = change_node_weight(std::move(graph), 3, 4);

  InitialCoarsener coarsener{&graph, create_default_context().coarsening};
  coarsener.TEST_mock_clustering({0, 1, 2, 3});

  auto [c_graph, c_mapping] = coarsener.TEST_contract_clustering();
  EXPECT_THAT(c_graph.n(), graph.n());
  EXPECT_THAT(c_graph.m(), graph.m());
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), graph.total_edge_weight());

  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 2));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 3));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(2, 4));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(3, 4));
}

TEST(InitialCoarsenerTest, ContractingAllNodesButOneWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};

  // 0--1
  // |  |
  // 2--3

  InitialCoarsener coarsener{&graph, create_default_context().coarsening};
  coarsener.TEST_mock_clustering({0, 1, 1, 1});
  auto [c_graph, c_mapping] = coarsener.TEST_contract_clustering();

  EXPECT_THAT(c_graph.n(), 2);
  EXPECT_THAT(c_graph.m(), 2); // one undirected edge
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), 2 * 2);
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 3));
}

TEST(InitialCoarsenerTest, ContractingGridHorizontallyWorks) {
  Graph graph{graphs::grid(2, 4)}; // two rows, 4 columns, organized row by row
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 2);
  graph = change_node_weight(std::move(graph), 2, 3);
  graph = change_node_weight(std::move(graph), 3, 4);
  graph = change_node_weight(std::move(graph), 4, 10);
  graph = change_node_weight(std::move(graph), 5, 20);
  graph = change_node_weight(std::move(graph), 6, 30);
  graph = change_node_weight(std::move(graph), 7, 40);

  InitialCoarsener coarsener{&graph, create_default_context().coarsening};
  coarsener.TEST_mock_clustering({0, 1, 2, 3, 0, 1, 2, 3});
  auto [c_graph, c_mapping] = coarsener.TEST_contract_clustering();

  EXPECT_THAT(c_graph.n(), 4);
  EXPECT_THAT(c_graph.m(), 2 * 3);
  EXPECT_THAT(c_graph.node_weights(), UnorderedElementsAre(11, 22, 33, 44));
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), 4 * 3);
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(11, 22));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(22, 33));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(33, 44));
}

TEST(InitialCoarsenerTest, ContractingGridVerticallyWorks) {
  Graph graph{graphs::grid(4, 2)}; // four columns, two rows, organized row by row
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 10);
  graph = change_node_weight(std::move(graph), 2, 2);
  graph = change_node_weight(std::move(graph), 3, 20);
  graph = change_node_weight(std::move(graph), 4, 3);
  graph = change_node_weight(std::move(graph), 5, 30);
  graph = change_node_weight(std::move(graph), 6, 4);
  graph = change_node_weight(std::move(graph), 7, 40);

  InitialCoarsener coarsener{&graph, create_default_context().coarsening};
  coarsener.TEST_mock_clustering({0, 0, 2, 2, 4, 4, 6, 6});
  auto [c_graph, c_mapping] = coarsener.TEST_contract_clustering();

  EXPECT_THAT(c_graph.n(), 4);
  EXPECT_THAT(c_graph.m(), 2 * 3);
  EXPECT_THAT(c_graph.node_weights(), UnorderedElementsAre(11, 22, 33, 44));
  EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
  EXPECT_THAT(c_graph.total_edge_weight(), 4 * 3);
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(11, 22));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(22, 33));
  EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(33, 44));
}
} // namespace kaminpar::ip
