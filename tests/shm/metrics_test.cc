#include <gmock/gmock.h>

#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/metrics.h"

namespace kaminpar::shm::testing {
class MetricsTestFixture : public ::testing::Test {
public:
  MetricsTestFixture()
      : graph(create_graph(
            {0, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 0, 0, 0, 0}, {4, 1, 1, 1, 1}, {3, 3, 3, 3, 3, 3, 3, 3}
        )) {}

  Graph graph;
};

TEST_F(MetricsTestFixture, parallel_bipartition_edge_cut) {
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 1, 1, 1, 1})};
  EXPECT_EQ(metrics::edge_cut(p_graph), 4 * 3);

  // star center to other block, should reduce the edge cut to 0
  p_graph.set_block(0, 1);
  EXPECT_EQ(metrics::edge_cut(p_graph), 0);

  // move center and two other nodes to block 1, should reduce the edge cut to 6
  for (NodeID u = 0; u < 3; ++u) {
    p_graph.set_block(u, 0);
  }
  EXPECT_EQ(metrics::edge_cut(p_graph), 2 * 3);
}

TEST_F(MetricsTestFixture, sequential_bipartition_edge_cut) {
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 1, 1, 1, 1})};
  EXPECT_EQ(metrics::edge_cut_seq(p_graph), 4 * 3);

  // star center to other block, should reduce the edge cut to 0
  p_graph.set_block(0, 1);
  EXPECT_EQ(metrics::edge_cut_seq(p_graph), 0);

  // move center and two other nodes to block 1, should reduce the edge cut to 6
  for (NodeID u = 0; u < 3; ++u) {
    p_graph.set_block(u, 0);
  }
  EXPECT_EQ(metrics::edge_cut_seq(p_graph), 2 * 3);
}

TEST_F(MetricsTestFixture, parallel_singleton_blocks_edge_cut) {
  PartitionedGraph p_graph{create_p_graph(graph, 5, {0, 1, 2, 3, 4})};
  EXPECT_EQ(metrics::edge_cut(p_graph), 4 * 3);
}

TEST_F(MetricsTestFixture, sequential_singleton_blocks_edge_cut) {
  PartitionedGraph p_graph{create_p_graph(graph, 5, {0, 1, 2, 3, 4})};
  EXPECT_EQ(metrics::edge_cut_seq(p_graph), 4 * 3);
}

TEST_F(MetricsTestFixture, perfectly_balanced_bipartition_balance) {
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 1, 1, 1, 1})};
  EXPECT_DOUBLE_EQ(metrics::imbalance(p_graph), 0.0);
}

TEST_F(MetricsTestFixture, imbalanced_bipartition_balance) {
  PartitionedGraph p_graph{create_p_graph(graph, 2, {0, 0, 0, 1, 1})};
  // block weights:
  // weight(0) = 6
  // weight(1) = 2
  // total weight: 8, avg: 4
  // --> imbalance: 50% = 0.5
  EXPECT_DOUBLE_EQ(metrics::imbalance(p_graph), 0.5);
}

inline Context
create_testing_context(const Graph &graph, const BlockID k = 2, const double epsilon = 0.03) {
  Context context;
  context.partition.k = k;
  context.partition.epsilon = epsilon;
  context.setup(graph);
  return context;
}

TEST(MetricsTest, is_feasible_with_single_node) {
  Graph graph{create_graph({0, 0}, {}, {1000}, {})};
  const PartitionedGraph p_graph{create_p_graph(graph, 1, {0})};
  Context ctx = create_testing_context(graph, 1, 0.03);

  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
}

TEST(MetricsTest, is_feasible_with_multiple_nodes) {
  Graph graph{create_graph({0, 0, 0, 0, 0}, {}, {200, 100, 100, 100}, {})};
  PartitionedGraph p_graph{create_p_graph(graph, 4, {0, 1, 2, 3})};
  Context ctx = create_testing_context(graph, 4, 0);

  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  p_graph.set_block(1, 0);
  p_graph.set_block(2, 0);
  EXPECT_FALSE(metrics::is_feasible(p_graph, ctx.partition));
}
} // namespace kaminpar::shm::testing
