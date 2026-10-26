#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/jet/jet_balancer.h"

namespace {
using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::testing;

using Cache = NormalDenseGainCache<CSRGraph>;
using Balancer = JetBalancer<CSRGraph, Cache>;

Graph weighted_empty_graph(const std::vector<NodeWeight> &weights) {
  return make_graph(std::vector<EdgeID>(weights.size() + 1, 0), {}, weights, {});
}

void verify_cache(const PartitionedGraph &p_graph, const Cache &cache) {
  for (const NodeID u : p_graph.graph().csr_graph().nodes()) {
    std::vector<EdgeWeight> connections(p_graph.k(), 0);
    p_graph.graph().csr_graph().adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      connections[p_graph.block(v)] += weight;
    });
    for (const BlockID b : p_graph.blocks()) {
      EXPECT_EQ(cache.conn(u, b), connections[b]);
    }
  }
  std::vector<BlockWeight> weights(p_graph.k(), 0);
  for (const NodeID u : p_graph.graph().csr_graph().nodes()) {
    weights[p_graph.block(u)] += p_graph.node_weight(u);
  }
  for (const BlockID b : p_graph.blocks()) {
    EXPECT_EQ(p_graph.block_weight(b), weights[b]);
  }
}

TEST(JetBalancerTest, LossBucketsFollowPaperOrdering) {
  EXPECT_EQ(jet::loss_bucket(-100), 0);
  EXPECT_EQ(jet::loss_bucket(-1), 0);
  EXPECT_EQ(jet::loss_bucket(0), 1);
  EXPECT_EQ(jet::loss_bucket(1), 2);
  EXPECT_EQ(jet::loss_bucket(2), 3);
  EXPECT_EQ(jet::loss_bucket(3), 3);
  EXPECT_EQ(jet::loss_bucket(4), 4);
  EXPECT_EQ(jet::loss_bucket(7), 4);
  EXPECT_EQ(jet::loss_bucket(8), 5);
  EXPECT_EQ(
      jet::loss_bucket(std::numeric_limits<EdgeWeight>::max()),
      std::numeric_limits<EdgeWeight>::digits + 1
  );
}

TEST(JetBalancerTest, LeavesBalancedPartitionUnchanged) {
  const auto graph = make_empty_graph(4);
  auto p_graph = make_p_graph(graph, 2, {0, 0, 1, 1});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, 2, 0.03);
  Cache cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_FALSE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_EQ(balancer.statistics().weak_rounds, 0);
  EXPECT_EQ(balancer.statistics().strong_rounds, 0);
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, WeakPassRebalancesDisconnectedVertices) {
  const auto graph = make_empty_graph(8);
  auto p_graph = make_p_graph(graph, 2, {0, 0, 0, 0, 0, 0, 1, 1});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, 2, 0.0);
  Cache cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_EQ(p_graph.block_weight(0), 4);
  EXPECT_EQ(p_graph.block_weight(1), 4);
  EXPECT_EQ(balancer.statistics().weak_rounds, 1);
  EXPECT_EQ(balancer.statistics().strong_rounds, 0);
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, WeakPassPrioritizesBeneficialMovesAndUpdatesCache) {
  const auto graph = make_graph(
      {0, 1, 2, 3, 4, 5, 6, 7, 8}, {7, 2, 1, 4, 3, 6, 5, 0}, {}, {8, 1, 1, 1, 1, 1, 1, 8}
  );
  auto p_graph = make_p_graph(graph, 2, {0, 0, 0, 0, 0, 0, 0, 1});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, 2, 0.25);
  Cache cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  ASSERT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  EXPECT_EQ(p_graph.block(0), 1);
  EXPECT_EQ(balancer.statistics().weak_rounds, 1);
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, StrongCookieCutterFillsTwoSmallDestinationsExactly) {
  const auto graph = make_empty_graph(6);
  auto p_graph = make_p_graph(graph, 3, {0, 0, 0, 0, 1, 2});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, 3, 0.0);
  Cache cache(ctx, graph.n(), 3);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_EQ(balancer.statistics().weak_rounds, 2);
  EXPECT_EQ(balancer.statistics().strong_rounds, 1);
  EXPECT_EQ(balancer.statistics().fallback_moves, 0);
  for (const BlockID b : p_graph.blocks()) {
    EXPECT_EQ(p_graph.block_weight(b), 2);
  }
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, HandlesNonuniformBounds) {
  const auto graph = make_empty_graph(8);
  auto p_graph = make_p_graph(graph, 3, {0, 0, 0, 0, 1, 1, 1, 2});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, std::vector<BlockWeight>{1, 3, 4});
  Cache cache(ctx, graph.n(), 3);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, FallbackHandlesVerticesExcludedByPaperWeightRestriction) {
  const auto graph = weighted_empty_graph({5, 5, 6});
  auto p_graph = make_p_graph(graph, 2, {0, 0, 1});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, std::vector<BlockWeight>{9, 14});
  Cache cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  EXPECT_EQ(balancer.statistics().fallback_moves, 1);
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, WeightedEvictionsRespectSourceLowerBound) {
  const auto graph = weighted_empty_graph({4, 3, 3});
  auto p_graph = make_p_graph(graph, 2, {0, 0, 1});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, std::vector<BlockWeight>{6, 6});
  ctx.partition.setup_min_block_weights(std::vector<BlockWeight>{4, 0});
  Cache cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  EXPECT_EQ(p_graph.block_weight(0), 4);
  EXPECT_EQ(p_graph.block(0), 0);
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, TerminatesOnInfeasibleWeightedPacking) {
  const auto graph = weighted_empty_graph({3, 3, 2});
  auto p_graph = make_p_graph(graph, 2, {0, 0, 1});
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, 2, 0.0);
  Cache cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache);
  EXPECT_FALSE(metrics::is_balanced(p_graph, ctx.partition));
  EXPECT_LE(balancer.statistics().strong_rounds, 2);
  verify_cache(p_graph, cache);
}

TEST(JetBalancerTest, RebalancesManyBlocksInParallel) {
  const auto graph = make_empty_graph(1024);
  auto p_graph = make_p_graph(graph, 17, std::vector<BlockID>(graph.n(), 0));
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.partition.setup(graph, 17, 0.03);
  Cache cache(ctx, graph.n(), 17);
  cache.initialize(graph.csr_graph(), p_graph);
  Balancer balancer(ctx);
  EXPECT_TRUE(balancer.refine(p_graph, graph.csr_graph(), ctx.partition, cache));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  verify_cache(p_graph, cache);
}

} // namespace
