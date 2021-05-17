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
using ::testing::UnorderedElementsAre;
using namespace ::kaminpar::test;

namespace kaminpar {
TEST(ParallelContractionTest, ContractingToSingleNodeWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};

  for (const bool leader_is_idempotent : {false, true}) {
    for (const NodeID cluster : {0, 1, 2, 3}) {
      auto [c_graph, c_mapping, m_ctx] = contract(graph, {cluster, cluster, cluster, cluster}, leader_is_idempotent);
      EXPECT_THAT(c_graph.n(), 1);
      EXPECT_THAT(c_graph.m(), 0);
      EXPECT_THAT(c_graph.node_weight(0), graph.total_node_weight());
    }
  }
}

TEST(ParallelContractionTest, ContractingToSingletonsWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};
  graph = change_node_weight(std::move(graph), 0, 1);
  graph = change_node_weight(std::move(graph), 1, 2);
  graph = change_node_weight(std::move(graph), 2, 3);
  graph = change_node_weight(std::move(graph), 3, 4);

  for (const bool leader_is_idempotent : {false, true}) {
    auto [c_graph, c_mapping, m_ctx] = contract(graph, {0, 1, 2, 3}, leader_is_idempotent);
    EXPECT_THAT(c_graph.n(), graph.n());
    EXPECT_THAT(c_graph.m(), graph.m());
    EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
    EXPECT_THAT(c_graph.total_edge_weight(), graph.total_edge_weight());

    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 2));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 3));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(2, 4));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(3, 4));
  }
}

TEST(ParallelContractionTest, ContractingAllNodesButOneWorks) {
  static constexpr auto GRID_LENGTH{2};
  Graph graph{graphs::grid(GRID_LENGTH, GRID_LENGTH)};

  // 0--1
  // |  |
  // 2--3

  for (const bool leader_is_idempotent : {false, true}) {
    auto [c_graph, c_mapping, m_ctx] = contract(graph, {0, 1, 1, 1}, leader_is_idempotent);
    EXPECT_THAT(c_graph.n(), 2);
    EXPECT_THAT(c_graph.m(), 2); // one undirected edge
    EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
    EXPECT_THAT(c_graph.total_edge_weight(), 2 * 2);
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(1, 3));
  }
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

  for (const bool leader_is_idempotent : {false, true}) {
    auto [c_graph, c_mapping, m_ctx] = contract(graph, {0, 1, 2, 3, 0, 1, 2, 3}, leader_is_idempotent);
    EXPECT_THAT(c_graph.n(), 4);
    EXPECT_THAT(c_graph.m(), 2 * 3);
    EXPECT_THAT(c_graph.node_weights(), UnorderedElementsAre(11, 22, 33, 44));
    EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
    EXPECT_THAT(c_graph.total_edge_weight(), 4 * 3);
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(11, 22));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(22, 33));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(33, 44));
  }
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

  for (const bool leader_is_idempotent : {false, true}) {
    auto [c_graph, c_mapping, m_ctx] = contract(graph, {0, 0, 2, 2, 4, 4, 6, 6}, leader_is_idempotent);
    EXPECT_THAT(c_graph.n(), 4);
    EXPECT_THAT(c_graph.m(), 2 * 3);
    EXPECT_THAT(c_graph.node_weights(), UnorderedElementsAre(11, 22, 33, 44));
    EXPECT_THAT(c_graph.total_node_weight(), graph.total_node_weight());
    EXPECT_THAT(c_graph.total_edge_weight(), 4 * 3);
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(11, 22));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(22, 33));
    EXPECT_THAT(c_graph, HasEdgeWithWeightedEndpoints(33, 44));
  }
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

  const auto permutations = sort_by_degree_buckets(nodes);
  const auto &permutation = permutations.old_to_new;
  EXPECT_THAT(permutation[0], AllOf(Ge(3), Le(4)));
  EXPECT_THAT(permutation[1], AllOf(Ge(1), Le(2)));
  EXPECT_THAT(permutation[2], Eq(5));
  EXPECT_THAT(permutation[3], AllOf(Ge(1), Le(2)));
  EXPECT_THAT(permutation[4], AllOf(Ge(3), Le(4)));
  EXPECT_THAT(permutation[5], Eq(0));
}

TEST(GraphPermutationTest, MovingIsolatedNodesToFrontWorks) {
  // node 0 1 2 3 4 5 6 7 8 9 10
  // deg  0 0 1 1 1 0 0 1 1 0 0
  const StaticArray<EdgeID> nodes = create_static_array<EdgeID>({0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5});
  const auto permutations = sort_by_degree_buckets(nodes, false);
  const auto &permutation = permutations.old_to_new;

  EXPECT_THAT(permutation[0], Le(5));
  EXPECT_THAT(permutation[1], Le(5));
  EXPECT_THAT(permutation[2], Ge(6));
  EXPECT_THAT(permutation[3], Ge(6));
  EXPECT_THAT(permutation[4], Ge(6));
  EXPECT_THAT(permutation[5], Le(5));
  EXPECT_THAT(permutation[6], Le(5));
  EXPECT_THAT(permutation[7], Ge(6));
  EXPECT_THAT(permutation[8], Ge(6));
  EXPECT_THAT(permutation[9], Le(5));
  EXPECT_THAT(permutation[10], Le(5));
}

TEST(GraphPermutationTest, MovingIsolatedNodesToBackWorks) {
  // node 0 1 2 3 4 5 6 7 8 9 10
  // deg  0 0 1 1 1 0 0 1 1 0 0
  const StaticArray<EdgeID> nodes = create_static_array<EdgeID>({0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5});
  const auto permutations = sort_by_degree_buckets(nodes, true);
  const auto &permutation = permutations.old_to_new;

  EXPECT_THAT(permutation[0], Ge(5));
  EXPECT_THAT(permutation[1], Ge(5));
  EXPECT_THAT(permutation[2], Le(4));
  EXPECT_THAT(permutation[3], Le(4));
  EXPECT_THAT(permutation[4], Le(4));
  EXPECT_THAT(permutation[5], Ge(5));
  EXPECT_THAT(permutation[6], Ge(5));
  EXPECT_THAT(permutation[7], Le(4));
  EXPECT_THAT(permutation[8], Le(4));
  EXPECT_THAT(permutation[9], Ge(5));
  EXPECT_THAT(permutation[10], Ge(5));
}

//
// Permutation
//

  /*
Graph build_permuted_graph_helper(const Graph &graph, const NodePermutation &permutation) {
  StaticArray<EdgeID> new_nodes(graph.n() + 1);
  StaticArray<NodeID> new_edges(graph.m());
  StaticArray<NodeWeight> new_node_weights(graph.n());
  StaticArray<EdgeWeight> new_edge_weights(graph.m());
  build_permuted_graph(graph.raw_nodes(), graph.raw_edges(), graph.raw_node_weights(), graph.raw_edge_weights(),
                       permutation, new_nodes, new_edges, new_node_weights, new_edge_weights);
  return Graph{std::move(new_nodes), std::move(new_edges), std::move(new_node_weights), std::move(new_edge_weights)};
}

TEST(GraphPermutationTest, PermutingEdgeWorks) {
  Graph graph{graphs::grid(2, 1)}; // single edge
  Graph permuted = build_permuted_graph_helper(graph, create_static_array<NodeID>({1, 0}));

  EXPECT_THAT(permuted.n(), Eq(graph.n()));
  EXPECT_THAT(permuted.m(), Eq(graph.m()));
  EXPECT_THAT(permuted, HasEdge(0, 1));
}

TEST(GraphPermutationTest, PermutingCompleteGraphWorks) {
  Graph graph{graphs::complete(5)};
  Graph permuted = build_permuted_graph_helper(graph, create_static_array<NodeID>({4, 3, 1, 0, 2}));

  EXPECT_THAT(permuted.n(), Eq(graph.n()));
  EXPECT_THAT(permuted.m(), Eq(graph.m()));
  for (NodeID u = 0; u < graph.n(); ++u) {
    for (NodeID v = u + 1; v < graph.n(); ++v) { EXPECT_THAT(permuted, HasEdge(u, v)); }
  }
}

TEST(GraphPermutationTest, PermutingPathWorks) {
  Graph graph{graphs::path(5)}; // 0 -- 1 -- 2 -- 3 -- 4
  Graph permuted = build_permuted_graph_helper(graph, create_static_array<NodeID>({4, 3, 1, 0, 2}));

  EXPECT_THAT(permuted.n(), Eq(graph.n()));
  EXPECT_THAT(permuted.m(), Eq(graph.m()));
  EXPECT_THAT(permuted, HasEdge(4, 3));
  EXPECT_THAT(permuted, HasEdge(3, 1));
  EXPECT_THAT(permuted, HasEdge(1, 0));
  EXPECT_THAT(permuted, HasEdge(0, 2));
}
  */

//
// Preprocessing
//

TEST(PreprocessingTest, PreprocessingFacadeRemovesIsolatedNodesAndAdaptsEpsilonFromUnweightedGraph) {
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
  const NodeWeight total_node_weight = 12;

  PartitionContext p_ctx;
  p_ctx.k = 2;
  p_ctx.epsilon = 0.17; // max block weight 7

  rearrange_and_remove_isolated_nodes(true, p_ctx, nodes, edges, node_weights, edge_weights, total_node_weight);

  EXPECT_THAT(nodes.size(), 7);
  EXPECT_THAT(edges.size(), 8);
  for (const NodeID v : edges) { EXPECT_THAT(v, Lt(7)); } // edges are valid

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
  Graph graph{test::create_graph({0, 2, 5, 6, 8, 11, 12}, {1, 3, 0, 4, 2, 1, 0, 4, 3, 1, 5, 4})};
  PartitionedGraph p_graph{test::create_p_graph(graph, 2, {0, 0, 0, 1, 1, 1})};

  SubgraphMemory memory{p_graph};
  SubgraphMemoryStartPosition position{0, 0};
  TemporarySubgraphMemory buffer{};
  const auto [subgraphs, positions] = extract_subgraphs_sequential(p_graph, position, memory, buffer);

  for (const auto &subgraph : subgraphs) {
    EXPECT_THAT(subgraph.n(), Eq(3));
    EXPECT_THAT(subgraph.m(), Eq(4));
    EXPECT_THAT(test::degrees(subgraph), UnorderedElementsAre(1, 1, 2));
  }

  EXPECT_THAT(positions[0].nodes_start_pos, Eq(0));
  EXPECT_THAT(positions[0].edges_start_pos, Eq(0));
  EXPECT_THAT(positions[1].nodes_start_pos, Eq(4));
  EXPECT_THAT(positions[1].edges_start_pos, Eq(4));
}

//
// Pseudo peripheral nodes
//
TEST(PseudoPeripheralTest, FindsPeripheralNodesInConnectedGraph) {
  Graph graph = test::graphs::grid(4, 4);
  const auto [u, v] = find_far_away_nodes<0>(graph, 1);
  EXPECT_THAT(u, Eq(0));
  EXPECT_THAT(v, Eq(15));
}

TEST(PseudoPeripheralTest, FindsPeripheralNodesInUnconnectedGraph) {
  Graph graph = test::graphs::empty(2);
  const auto [u, v] = find_far_away_nodes<0>(graph, 1);
  EXPECT_THAT(u, Eq(0));
  EXPECT_THAT(v, Eq(1));
}

TEST(PseudoPeripheralTest, FindsPeripheralNodesInLargerUnconnectedGraph) {
  Graph block0 = test::graphs::grid(16, 16);
  Graph block1 = test::graphs::empty(1);
  Graph graph = test::merge_graphs({&block0, &block1});
  const auto [u, v] = find_far_away_nodes<0>(graph, 1);
  EXPECT_THAT(u, Eq(0));
  EXPECT_THAT(v, Eq(block0.n())); // first node in block1
}
} // namespace kaminpar
