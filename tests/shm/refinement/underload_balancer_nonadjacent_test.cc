#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/underload_balancer.h"

namespace kaminpar::shm::testing {
namespace {

void expect_consistent_bounds(const PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  std::vector<BlockWeight> weights(p_graph.k(), 0);
  const auto &graph = p_graph.graph().csr_graph();
  for (const NodeID node : graph.nodes()) {
    ASSERT_LT(p_graph.block(node), p_graph.k());
    weights[p_graph.block(node)] += graph.node_weight(node);
  }
  for (const BlockID block : p_graph.blocks()) {
    EXPECT_EQ(p_graph.block_weight(block), weights[block]);
    EXPECT_GE(weights[block], p_ctx.min_block_weight(block));
    EXPECT_LE(weights[block], p_ctx.max_block_weight(block));
  }
}

TEST(UnderloadBalancerNonadjacentTest, FillsAnEmptyBlockWithoutAdjacentDonors) {
  const Graph graph = make_complete_graph(12);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {6, 6, 6});
  ctx.partition.setup_min_block_weights({4, 4, 4});
  auto p_graph = make_p_graph(graph, 3, {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1});
  UnderloadBalancer balancer(ctx);

  EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));

  expect_consistent_bounds(p_graph, ctx.partition);
  EXPECT_EQ(p_graph.block_weight(2), 4);
}

TEST(UnderloadBalancerNonadjacentTest, FillsDisconnectedCommunitiesFromRemoteSlack) {
  GraphBuilder builder;
  for (const NodeID size : {8, 6, 2}) {
    const NodeID first = builder.new_node();
    builder.new_edge(first + size - 1);
    builder.new_edge(first + 1);
    for (NodeID offset = 1; offset < size; ++offset) {
      builder.new_node();
      builder.new_edge(first + offset - 1);
      builder.new_edge(first + (offset + 1) % size);
    }
  }
  const Graph graph = builder.build();
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {9, 8, 7});
  ctx.partition.setup_min_block_weights({5, 5, 5});
  auto p_graph = make_p_graph(graph, 3, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2});
  UnderloadBalancer balancer(ctx);

  EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));

  expect_consistent_bounds(p_graph, ctx.partition);
  EXPECT_EQ(p_graph.block_weight(2), 5);
}

TEST(UnderloadBalancerNonadjacentTest, ExplicitUnitWeightsSkipZeroWeightNodes) {
  const Graph graph = make_graph(std::vector<EdgeID>(9, 0), {}, {0, 1, 1, 1, 0, 1, 1, 1}, {});
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {3, 3, 2});
  ctx.partition.setup_min_block_weights({2, 2, 2});
  auto p_graph = make_p_graph(graph, 3, {0, 0, 0, 0, 1, 1, 1, 1});
  UnderloadBalancer balancer(ctx);

  EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));

  expect_consistent_bounds(p_graph, ctx.partition);
  EXPECT_EQ(p_graph.block(0), 0);
  EXPECT_EQ(p_graph.block(4), 1);
}

TEST(UnderloadBalancerNonadjacentTest, WeightedNodesFitTargetsAndPreserveDonorMinima) {
  const Graph graph = make_graph(std::vector<EdgeID>(7, 0), {}, {3, 1, 1, 1, 1, 1}, {});
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {5, 2, 5});
  ctx.partition.setup_min_block_weights({2, 2, 4});
  auto p_graph = make_p_graph(graph, 3, {0, 0, 0, 1, 2, 2});
  UnderloadBalancer balancer(ctx);

  EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));

  expect_consistent_bounds(p_graph, ctx.partition);
  EXPECT_EQ(p_graph.block(0), 2);
  EXPECT_EQ(p_graph.block_weight(0), 2);
  EXPECT_EQ(p_graph.block_weight(1), 2);
  EXPECT_EQ(p_graph.block_weight(2), 4);
}

TEST(UnderloadBalancerNonadjacentTest, InsufficientSlackDoesNotUnderloadDonors) {
  const Graph graph = make_empty_graph(4);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {4, 4});
  ctx.partition.setup_min_block_weights({3, 3});
  auto p_graph = make_p_graph(graph, 2, {0, 0, 0, 0});
  UnderloadBalancer balancer(ctx);

  EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));

  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 1);
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  EXPECT_FALSE(metrics::is_min_balanced(p_graph, ctx.partition));
}

TEST(UnderloadBalancerNonadjacentTest, UnconfiguredMinimaLeaveThePartitionUnchanged) {
  const Graph graph = make_complete_graph(12);
  const std::vector<BlockID> partition(12, 0);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {12, 12, 12});
  auto p_graph = make_p_graph(graph, 3, partition);
  UnderloadBalancer balancer(ctx);

  EXPECT_FALSE(balancer.refine(p_graph, ctx.partition));

  for (const NodeID node : graph.csr_graph().nodes()) {
    EXPECT_EQ(p_graph.block(node), partition[node]);
  }
}

TEST(UnderloadBalancerNonadjacentTest, CompletesAfterParallelAdjacentMovesOnRepeatedRefinement) {
  const Graph graph = make_complete_graph(128);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {128, 44, 44});
  ctx.partition.setup_min_block_weights({42, 42, 42});
  UnderloadBalancer balancer(ctx);
  tbb::task_arena arena(4);
  for (int repetition = 0; repetition < 8; ++repetition) {
    std::vector<BlockID> partition(128, 0);
    std::fill(partition.end() - 28, partition.end(), 1);
    auto p_graph = make_p_graph(graph, 3, partition);

    arena.execute([&] {
      balancer.initialize(p_graph);
      EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));
      EXPECT_FALSE(balancer.refine(p_graph, ctx.partition));
    });

    expect_consistent_bounds(p_graph, ctx.partition);
    EXPECT_EQ(p_graph.block_weight(0) + p_graph.block_weight(1), 86);
    EXPECT_EQ(p_graph.block_weight(2), 42);
  }
}

} // namespace
} // namespace kaminpar::shm::testing
