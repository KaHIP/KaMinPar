/***********************************************************************************************************************
 * @file:   greedy_node_coloring_test.cc
 * @author: Daniel Seemaier
 * @date:   11.11.2022
 * @brief:  Unit tests for the greedy node (vertex) coloring algorithm.
 **********************************************************************************************************************/
#include <limits>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tests/dist/distributed_graph_factories.h"
#include "tests/dist/distributed_graph_helpers.h"

#include "kaminpar-mpi/utils.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

namespace {
template <typename Coloring>
void validate_node_coloring(
    const DistributedGraph &graph,
    const Coloring &coloring,
    const ColorID max_num_colors = std::numeric_limits<ColorID>::max()
) {
  ASSERT_GE(coloring.size(), graph.total_n());
  for (const NodeID u : graph.nodes()) {
    EXPECT_LT(coloring[u], max_num_colors);
    for (const NodeID v : graph.adjacent_nodes(u)) {
      EXPECT_NE(coloring[u], coloring[v]);
    }
  }
}
} // namespace

TEST(GreedyNodeColoringTest, colors_empty_graph) {
  auto graph = make_empty_graph();
  auto coloring = compute_node_coloring_sequentially(graph, 1);
  EXPECT_TRUE(coloring.empty());
}

TEST(GreedyNodeColoringTest, colors_isolated_nodes_graph) {
  constexpr NodeID kNodesPerPE = 2;
  constexpr NodeID kNumberOfSupersteps = 1;

  const auto graph = make_isolated_nodes_graph(kNodesPerPE);
  const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
  validate_node_coloring(graph, coloring, 1);
}

TEST(GreedyNodeColoringTest, colors_isolated_edges_graph) {
  constexpr NodeID kNodesPerPE = 2;
  constexpr NodeID kNumberOfSupersteps = 1;

  const auto graph = make_isolated_edges_graph(kNodesPerPE);
  const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
  validate_node_coloring(graph, coloring, 2);
}

TEST(GreedyNodeColoringTest, colors_circle_graph) {
  constexpr NodeID kNumberOfSupersteps = 1;
  const auto graph = make_circle_graph();
  const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
  validate_node_coloring(graph, coloring, 3);
}

TEST(GreedyNodeColoringTest, colors_circle_clique_graph_2) {
  constexpr NodeID kNumberOfNodesPerPE = 2;
  constexpr NodeID kNumberOfSupersteps = 1;
  const auto graph = make_circle_clique_graph(kNumberOfNodesPerPE);
  const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
  validate_node_coloring(graph, coloring);
}

TEST(GreedyNodeColoringTest, colors_circle_clique_graph_5) {
  constexpr NodeID kNumberOfNodesPerPE = 5;
  constexpr NodeID kNumberOfSupersteps = 1;
  const auto graph = make_circle_clique_graph(kNumberOfNodesPerPE);
  const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
  validate_node_coloring(graph, coloring);
}

TEST(GreedyNodeColoringTest, colors_circle_clique_graph_5_many_steps) {
  constexpr NodeID kNumberOfNodesPerPE = 5;
  constexpr NodeID kNumberOfSupersteps = 5;
  const auto graph = make_circle_clique_graph(kNumberOfNodesPerPE);
  const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
  validate_node_coloring(graph, coloring);
}
} // namespace kaminpar::dist
