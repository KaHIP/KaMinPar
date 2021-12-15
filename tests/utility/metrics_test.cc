#include "datastructure/graph.h"
#include "tests.h"
#include "utility/metrics.h"

#include "gmock/gmock.h"

using ::testing::DoubleEq;
using ::testing::Eq;
using ::testing::Test;

namespace kaminpar::metrics {
class AWeightedStar : public Test {
public:
  AWeightedStar()
      : graph(test::create_graph({0, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 0, 0, 0, 0}, {4, 1, 1, 1, 1},
                                 {3, 3, 3, 3, 3, 3, 3, 3})) {}

  Graph graph;
};

TEST_F(AWeightedStar, BipartitionEdgeCutIsCorrect) {
  PartitionedGraph p_graph{test::create_p_graph(graph, 2, {0, 1, 1, 1, 1})};
  ASSERT_THAT(metrics::edge_cut(p_graph), Eq(4 * 3));

  // star center to other block, should reduce the edge cut to 0
  p_graph.set_block(0, 1);
  ASSERT_THAT(metrics::edge_cut(p_graph), Eq(0));

  // move center and two other nodes to block 1, should reduce the edge cut to 6
  for (NodeID u = 0; u < 3; ++u) p_graph.set_block(u, 0);
  ASSERT_THAT(metrics::edge_cut(p_graph), Eq(2 * 3));
}

TEST_F(AWeightedStar, BipartitionSeqEdgeCutIsCorrect) {
  PartitionedGraph p_graph{test::create_p_graph(graph, 2, {0, 1, 1, 1, 1})};
  ASSERT_THAT(metrics::edge_cut(p_graph, tag::seq), Eq(4 * 3));

  // star center to other block, should reduce the edge cut to 0
  p_graph.set_block(0, 1);
  ASSERT_THAT(metrics::edge_cut(p_graph, tag::seq), Eq(0));

  // move center and two other nodes to block 1, should reduce the edge cut to 6
  for (NodeID u = 0; u < 3; ++u) p_graph.set_block(u, 0);
  ASSERT_THAT(metrics::edge_cut(p_graph, tag::seq), Eq(2 * 3));
}

TEST_F(AWeightedStar, FiveWayEdgeCutIsCorrect) {
  PartitionedGraph p_graph{test::create_p_graph(graph, 5, {0, 1, 2, 3, 4})};
  ASSERT_THAT(metrics::edge_cut(p_graph), Eq(4 * 3));
}

TEST_F(AWeightedStar, FiveWaySeqEdgeCutIsCorrect) {
  PartitionedGraph p_graph{test::create_p_graph(graph, 5, {0, 1, 2, 3, 4})};
  ASSERT_THAT(metrics::edge_cut(p_graph, tag::seq), Eq(4 * 3));
}

TEST_F(AWeightedStar, PerfectlyBalancedBipartitionHasImbalanceZero) {
  PartitionedGraph p_graph{test::create_p_graph(graph, 2, {0, 1, 1, 1, 1})};
  ASSERT_THAT(metrics::imbalance(p_graph), DoubleEq(0.0));
}

TEST_F(AWeightedStar, ImbalancedBipartitionHasCorrectImbalance) {
  PartitionedGraph p_graph{test::create_p_graph(graph, 2, {0, 0, 0, 1, 1})};
  // block weights:
  // weight(0) = 6
  // weight(1) = 2
  // total weight: 8, avg: 4
  // --> imbalance: 50% = 0.5
  ASSERT_THAT(metrics::imbalance(p_graph), DoubleEq(0.5));
}

TEST(MetricsTest, IsFeasibleMetricWorksForGraphWithSingleNode) {
  Graph graph{test::create_graph({0, 0}, {}, {1000}, {})};
  const PartitionedGraph p_graph{test::create_p_graph(graph, 1, {0})};
  Context ctx = create_default_context(graph, 1, 0.03);

  ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
}

TEST(MetricsTest, IsFeasibleMetricWorks) {
  Graph graph{test::create_graph({0, 0, 0, 0, 0}, {}, {200, 100, 100, 100}, {})};
  PartitionedGraph p_graph{test::create_p_graph(graph, 4, {0, 1, 2, 3})};
  Context ctx = create_default_context(graph, 4, 0);

  ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  p_graph.set_block(1, 0);
  p_graph.set_block(2, 0);
  ASSERT_FALSE(metrics::is_feasible(p_graph, ctx.partition));
}
} // namespace kaminpar::metrics