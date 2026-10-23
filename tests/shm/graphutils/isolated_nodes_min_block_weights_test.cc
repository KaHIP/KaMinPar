#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/metrics.h"

namespace kaminpar::shm::testing {
namespace {

Graph make_cycle_with_isolated_nodes(const NodeID cycle_nodes, const NodeID isolated_nodes) {
  GraphBuilder builder;
  for (NodeID u = 0; u < cycle_nodes; ++u) {
    builder.new_node();
    builder.new_edge((u + cycle_nodes - 1) % cycle_nodes);
    builder.new_edge((u + 1) % cycle_nodes);
  }
  for (NodeID u = 0; u < isolated_nodes; ++u) {
    builder.new_node();
  }
  return builder.build(true);
}

PartitionedGraph integrate_isolated_nodes(
    Graph &graph,
    const NodeID num_isolated_nodes,
    const std::vector<BlockID> &active_partition,
    const PartitionContext &p_ctx
) {
  graph.csr_graph().remove_isolated_nodes(num_isolated_nodes);
  auto p_graph = make_p_graph(graph, p_ctx.k, active_partition);
  EXPECT_EQ(graph.csr_graph().integrate_isolated_nodes(), num_isolated_nodes);
  return graph::assign_isolated_nodes(std::move(p_graph), num_isolated_nodes, p_ctx);
}

TEST(IsolatedNodesMinBlockWeightsTest, ReservesWeightProportionallyAcrossHeterogeneousMinima) {
  Graph graph = make_cycle_with_isolated_nodes(15, 7);
  PartitionContext ctx;
  ctx.setup(graph, {22, 22, 22, 22});
  ctx.setup_min_block_weights({0, 4, 7, 9});

  EXPECT_EQ(
      graph::compute_nonisolated_min_block_weights(ctx, 7), std::vector<BlockWeight>({0, 3, 5, 5})
  );
  EXPECT_EQ(
      graph::compute_nonisolated_min_block_weights(ctx, 0), std::vector<BlockWeight>({0, 4, 7, 9})
  );
  EXPECT_EQ(
      graph::compute_nonisolated_min_block_weights(ctx, 22), std::vector<BlockWeight>({0, 0, 0, 0})
  );
}

TEST(IsolatedNodesMinBlockWeightsTest, ReservationArithmeticDoesNotOverflowBlockWeights) {
  Graph graph = make_cycle_with_isolated_nodes(4, 0);
  PartitionContext ctx;
  const BlockWeight minimum = std::numeric_limits<BlockWeight>::max() / 4;
  ctx.setup(graph, {minimum, minimum, minimum, minimum});
  ctx.setup_min_block_weights({minimum, minimum, minimum, minimum});
  const BlockWeight reserved = 3 * minimum + 1;

  const auto active_minima = graph::compute_nonisolated_min_block_weights(ctx, reserved);

  EXPECT_EQ(
      std::accumulate(active_minima.begin(), active_minima.end(), BlockWeight{0}),
      4 * minimum - reserved
  );
  for (const BlockWeight active_minimum : active_minima) {
    EXPECT_GE(active_minimum, 0);
    EXPECT_LE(active_minimum, minimum);
  }
}

TEST(IsolatedNodesMinBlockWeightsTest, HandlesUnconfiguredAndZeroMinima) {
  Graph graph = make_cycle_with_isolated_nodes(4, 4);
  PartitionContext ctx;
  ctx.setup(graph, {4, 4});
  EXPECT_EQ(graph::compute_nonisolated_min_block_weights(ctx, 4), std::vector<BlockWeight>({0, 0}));
  ctx.setup_min_block_weights({0, 0});
  EXPECT_EQ(graph::compute_nonisolated_min_block_weights(ctx, 4), std::vector<BlockWeight>({0, 0}));
}

TEST(IsolatedNodesMinBlockWeightsTest, FillsEveryMinimumDeficitBeforeSurplusCapacity) {
  Graph graph = make_cycle_with_isolated_nodes(24, 16);
  PartitionContext ctx;
  ctx.setup(graph, {11, 11, 11, 11});
  ctx.setup_min_block_weights({9, 9, 9, 9});
  EXPECT_EQ(
      graph::compute_nonisolated_min_block_weights(ctx, 16), std::vector<BlockWeight>({5, 5, 5, 5})
  );
  std::vector<BlockID> active_partition(24);
  for (NodeID u = 0; u < 24; ++u) {
    active_partition[u] = u / 6;
  }

  const auto p_graph = integrate_isolated_nodes(graph, 16, active_partition, ctx);

  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx));
  for (NodeID u = 0; u < 24; ++u) {
    EXPECT_EQ(p_graph.block(u), active_partition[u]);
  }
}

TEST(IsolatedNodesMinBlockWeightsTest, PreservesAssignmentOrderWithoutMinimumBounds) {
  Graph graph = make_cycle_with_isolated_nodes(24, 16);
  PartitionContext ctx;
  ctx.setup(graph, {11, 11, 11, 11});
  std::vector<BlockID> active_partition(24);
  for (NodeID u = 0; u < 24; ++u) {
    active_partition[u] = u / 6;
  }

  const auto p_graph = integrate_isolated_nodes(graph, 16, active_partition, ctx);

  EXPECT_EQ(p_graph.block_weight(0), 11);
  EXPECT_EQ(p_graph.block_weight(1), 11);
  EXPECT_EQ(p_graph.block_weight(2), 11);
  EXPECT_EQ(p_graph.block_weight(3), 7);
}

TEST(IsolatedNodesMinBlockWeightsTest, AssignsAllIsolatedUnitNodesWithinBounds) {
  Graph graph = make_cycle_with_isolated_nodes(0, 12);
  PartitionContext ctx;
  ctx.setup(graph, {5, 5, 5});
  ctx.setup_min_block_weights({3, 4, 3});
  EXPECT_EQ(
      graph::compute_nonisolated_min_block_weights(ctx, 12), std::vector<BlockWeight>({0, 0, 0})
  );

  const auto p_graph = integrate_isolated_nodes(graph, 12, {}, ctx);

  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx));
  EXPECT_EQ(p_graph.block_weight(0), 5);
  EXPECT_EQ(p_graph.block_weight(1), 4);
  EXPECT_EQ(p_graph.block_weight(2), 3);
}

TEST(IsolatedNodesMinBlockWeightsTest, WeightedNodesSkipUnderloadedBlocksThatCannotFit) {
  Graph graph = make_graph({0, 1, 2, 2, 2}, {1, 0}, {5, 2, 4, 1}, {1, 1}, true);
  PartitionContext ctx;
  ctx.setup(graph, {6, 6});
  ctx.setup_min_block_weights({6, 6});

  const auto p_graph = integrate_isolated_nodes(graph, 2, {0, 1}, ctx);

  EXPECT_EQ(p_graph.block(2), 1);
  EXPECT_EQ(p_graph.block(3), 0);
  EXPECT_EQ(p_graph.block_weight(0), 6);
  EXPECT_EQ(p_graph.block_weight(1), 6);
  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx));
}

TEST(IsolatedNodesMinBlockWeightsTest, ReadsRestoredWeightsWhenEveryNodeWasIsolated) {
  Graph graph = make_graph({0, 0, 0, 0, 0}, {}, {4, 1, 3, 2}, {}, true);
  PartitionContext ctx;
  ctx.setup(graph, {6, 6});
  ctx.setup_min_block_weights({4, 4});

  const auto p_graph = integrate_isolated_nodes(graph, 4, {}, ctx);

  EXPECT_EQ(p_graph.block_weight(0) + p_graph.block_weight(1), 10);
  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx));
}

TEST(IsolatedNodesMinBlockWeightsTest, PublicPartitionerRestoresOriginalBoundsAndNodeCounts) {
  for (Context ctx : {create_default_context(), create_fast_context(), create_eco_context()}) {
    Graph graph = make_cycle_with_isolated_nodes(24, 16);
    KaMinPar partitioner(2, std::move(ctx));
    partitioner.set_output_level(OutputLevel::QUIET);
    partitioner.set_graph(std::move(graph));
    partitioner.set_k(4);
    partitioner.set_absolute_max_block_weights(std::vector<BlockWeight>{11, 11, 11, 11});
    partitioner.set_absolute_min_block_weights(std::vector<BlockWeight>{9, 9, 9, 9});
    std::vector<BlockID> partition(40);

    partitioner.compute_partition(partition);

    std::vector<BlockWeight> weights(4, 0);
    for (const BlockID block : partition) {
      ASSERT_LT(block, 4);
      ++weights[block];
    }
    for (BlockID b = 0; b < 4; ++b) {
      EXPECT_GE(weights[b], 9);
      EXPECT_LE(weights[b], 11);
      EXPECT_EQ(partitioner.context().partition.min_block_weight(b), 9);
    }
    EXPECT_EQ(partitioner.context().partition.n, 40);
    EXPECT_EQ(partitioner.context().partition.total_node_weight, 40);
  }
}

} // namespace
} // namespace kaminpar::shm::testing
