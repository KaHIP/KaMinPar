/*******************************************************************************
 * Integration tests for Jet cache and rebalancer combinations.
 ******************************************************************************/
#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/jet/jet_refiner.h"

namespace kaminpar::shm {
namespace {
using namespace testing;

void refine_and_check(Graph &graph, const int threads) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
  for (const auto cache : {JetGainCacheStrategy::SPARSE, JetGainCacheStrategy::JET}) {
    for (const auto balancer : {JetBalancingAlgorithm::OVERLOAD, JetBalancingAlgorithm::JET}) {
      auto ctx = create_jet_context(2);
      ctx.parallel.num_threads = threads;
      ctx.refinement.jet.gain_cache_strategy = cache;
      ctx.refinement.jet.balancing_algorithm = balancer;
      ctx.refinement.jet.num_iterations = 10;
      ctx.refinement.jet.num_fruitless_iterations = 3;
      ctx.partition.setup(graph, 8, 0.1);
      std::vector<BlockID> partition(graph.n());
      for (NodeID u = 0; u < graph.n(); ++u) {
        partition[u] = u % 8;
      }
      auto p_graph = make_p_graph(graph, 8, partition);
      const EdgeWeight before = metrics::edge_cut(p_graph);
      JetRefiner refiner(ctx);
      refiner.initialize(p_graph);
      refiner.refine(p_graph, ctx.partition);
      EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
      EXPECT_LE(metrics::edge_cut(p_graph), before);
      std::vector<BlockWeight> weights(8, 0);
      for (NodeID u = 0; u < graph.n(); ++u) {
        ASSERT_LT(p_graph.block(u), 8);
        weights[p_graph.block(u)] += p_graph.node_weight(u);
      }
      for (BlockID b = 0; b < 8; ++b) {
        EXPECT_EQ(p_graph.block_weight(b), weights[b]);
      }
    }
  }
}

TEST(JetRefinerTest, AllCombinationsKeepFeasibleBestOnCSR) {
  auto graph = make_grid_graph(16, 16);
  refine_and_check(graph, 1);
  refine_and_check(graph, 4);
}

TEST(JetRefinerTest, AllCombinationsKeepFeasibleBestOnCompressedGraph) {
  auto csr = make_grid_graph(16, 16);
  auto graph = compress(csr.csr_graph());
  refine_and_check(graph, 4);
}

TEST(JetRefinerTest, AdaptiveCachePreservesLegacyMovesWithOneThread) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
  auto graph = make_grid_graph(12, 12);
  std::vector<BlockID> partition(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    partition[u] = u % 8;
  }
  std::vector<BlockID> reference;
  for (const auto cache : {JetGainCacheStrategy::SPARSE, JetGainCacheStrategy::JET}) {
    auto ctx = create_jet_context(2);
    ctx.refinement.jet.gain_cache_strategy = cache;
    ctx.refinement.jet.balancing_algorithm = JetBalancingAlgorithm::OVERLOAD;
    ctx.refinement.jet.num_iterations = 10;
    ctx.partition.setup(graph, 8, 0.1);
    auto p_graph = make_p_graph(graph, 8, partition);
    KaMinPar::reseed(42);
    JetRefiner refiner(ctx);
    refiner.initialize(p_graph);
    refiner.refine(p_graph, ctx.partition);
    std::vector<BlockID> result(graph.n());
    for (NodeID u = 0; u < graph.n(); ++u) {
      result[u] = p_graph.block(u);
    }
    if (reference.empty()) {
      reference = result;
    } else {
      EXPECT_EQ(result, reference);
    }
  }
}

TEST(JetRefinerTest, DoesNotRollbackToInfeasibleLowerCutInput) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
  auto graph = make_path_graph(64);
  auto ctx = create_jet_context(1);
  ctx.partition.setup(graph, 4, 0.1);
  ctx.refinement.jet.num_iterations = 3;
  auto p_graph = make_p_graph(graph, 4, std::vector<BlockID>(graph.n(), 0));
  ASSERT_EQ(metrics::edge_cut(p_graph), 0);
  ASSERT_FALSE(metrics::is_balanced(p_graph, ctx.partition));
  JetRefiner refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
}

TEST(JetRefinerTest, RetainsBalanceProgressWhenWeightedPackingIsImpossible) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
  auto graph = make_graph({0, 0, 0, 0}, {}, {9, 9, 9}, {});
  auto ctx = create_jet_context(1);
  ctx.partition.setup(graph, std::vector<BlockWeight>{14, 14});
  ctx.refinement.jet.num_iterations = 3;
  auto p_graph = make_p_graph(graph, 2, {0, 0, 0});
  const BlockWeight initial_overload = metrics::total_overload(p_graph, ctx.partition);
  JetRefiner refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);
  EXPECT_LT(metrics::total_overload(p_graph, ctx.partition), initial_overload);
  EXPECT_EQ(metrics::total_overload(p_graph, ctx.partition), 4);
}

TEST(JetRefinerTest, PreservesConfiguredMinimumWeights) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
  auto graph = make_grid_graph(12, 12);
  auto ctx = create_jet_context(2);
  ctx.partition.setup(graph, 8, 0.1);
  ctx.partition.setup_min_block_weights(std::vector<BlockWeight>(8, 17));
  ctx.refinement.jet.num_iterations = 5;
  std::vector<BlockID> partition(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    partition[u] = u % 8;
  }
  for (const auto balancer : {JetBalancingAlgorithm::OVERLOAD, JetBalancingAlgorithm::JET}) {
    ctx.refinement.jet.balancing_algorithm = balancer;
    auto p_graph = make_p_graph(graph, 8, partition);
    JetRefiner refiner(ctx);
    refiner.initialize(p_graph);
    refiner.refine(p_graph, ctx.partition);
    EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
    EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
  }
}

} // namespace
} // namespace kaminpar::shm
