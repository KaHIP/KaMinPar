/*******************************************************************************
 * Regression tests for concurrent FM commits and per-round configuration.
 ******************************************************************************/
#include <array>
#include <numeric>
#include <random>

#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"
#include "kaminpar-shm/refinement/fm/border_nodes.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"
#include "kaminpar-shm/refinement/fm/unconstrained_fm_refiner.h"
#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::testing {
namespace {

void check_weights(const PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  std::vector<BlockWeight> weights(p_graph.k());
  for (NodeID u = 0; u < p_graph.n(); ++u) {
    weights[p_graph.block(u)] += p_graph.node_weight(u);
  }
  for (const BlockID b : p_graph.blocks()) {
    EXPECT_EQ(p_graph.block_weight(b), weights[b]);
    EXPECT_LE(weights[b], p_ctx.max_block_weight(b));
  }
}

TEST(FMBorderNodesTest, ReinitializationDiscardsOldNodesAndResetsOwnership) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 8);
  constexpr NodeID n = 256;
  EdgeBasedGraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) {
    builder.add_edge(u, (u + 1) % n);
  }
  const Graph graph = builder.build();
  std::vector<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    partition[u] = u % 2;
  }
  auto p_graph = make_p_graph(graph, 2, partition);
  Context ctx = create_default_context();
  ctx.partition.setup(graph, 2, .03);
  NormalCompactHashingGainCache<CSRGraph> cache(ctx, n, 2);
  fm::NodeTracker tracker(n);
  fm::BorderNodes border(cache, tracker);
  cache.initialize(graph.csr_graph(), p_graph);
  border.init(p_graph);
  ASSERT_EQ(border.size(), n);
  EXPECT_EQ(border.poll(n, 1, [](NodeID) {}), n);

  // All former border nodes become internal; cached per-thread buffers must be empty.
  for (NodeID u = 0; u < n; ++u) {
    p_graph.set_block(u, 0);
  }
  cache.initialize(graph.csr_graph(), p_graph);
  border.init(p_graph);
  EXPECT_EQ(border.size(), 0u);
  EXPECT_FALSE(border.has_more());

  EdgeBasedGraphBuilder small_builder;
  small_builder.add_edge(0, 1);
  const Graph small = small_builder.build();
  auto small_partition = make_p_graph(small, 2, {0, 1});
  ctx.partition.setup(small, 2, .03);
  cache.initialize(small.csr_graph(), small_partition);
  border.init(small_partition);
  ASSERT_EQ(border.size(), 2u);
  std::array<unsigned, 2> visits{};
  EXPECT_EQ(
      border.poll(
          3,
          1,
          [&](const NodeID u) {
            ASSERT_LT(u, visits.size());
            ++visits[u];
          }
      ),
      2u
  );
  EXPECT_EQ(visits, (std::array<unsigned, 2>{1, 1}));

  border.init_precomputed(small_partition, std::array<NodeID, 1>{1});
  EXPECT_EQ(border.poll(2, 1, [](const NodeID u) { EXPECT_EQ(u, 1u); }), 1u);
  border.init(small_partition);
  EXPECT_EQ(border.size(), 2u);
  EXPECT_EQ(border.poll(3, 1, [](NodeID) {}), 2u);
}

TEST(FMCorrectnessTest, ConcurrentFMAndBalancerProduceFeasiblePartitions) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 8);
  tbb::task_arena arena(8);
  std::mt19937 rng(501);
  for (int repetition = 0; repetition < 100; ++repetition) {
    SCOPED_TRACE(repetition);
    constexpr NodeID n = 128;
    const BlockID k = std::array<BlockID, 3>{4, 8, 32}[repetition % 3];
    EdgeBasedGraphBuilder builder;
    // A cycle makes all node IDs present, independently of the random edges.
    for (NodeID u = 0; u < n; ++u) {
      builder.add_edge(u, (u + 1) % n);
      for (NodeID v = u + 2; v < n; ++v) {
        if (!(u == 0 && v == n - 1) && rng() % 100 < 8) {
          builder.add_edge(u, v, 1 + rng() % 20);
        }
      }
    }
    Graph graph = builder.build();
    std::vector<BlockID> partition(n);
    for (NodeID u = 0; u < n; ++u) {
      partition[u] = u % k;
    }
    auto p_graph = make_p_graph(graph, k, partition);
    Context ctx = create_default_context();
    ctx.parallel.num_threads = 8;
    ctx.partition.setup(graph, std::vector<BlockWeight>(k, n / k + 1));
    ctx.refinement.kway_fm.num_iterations = 3;
    ctx.refinement.kway_fm.num_seed_nodes = 1;
    ctx.refinement.kway_fm.minimal_parallelism = 8;
    ctx.refinement.kway_fm.gain_cache_strategy = std::array{
        GainCacheStrategy::COMPACT_HASHING,
        GainCacheStrategy::SPARSE,
        GainCacheStrategy::ON_THE_FLY,
        GainCacheStrategy::AUTO
    }[(repetition / 3) % 4];
    FMRefiner refiner(ctx);
    refiner.initialize(p_graph);
    arena.execute([&] { refiner.refine(p_graph, ctx.partition); });
    // Concurrent FM commits may briefly exceed upper bounds. The refinement pipeline
    // follows FM with the balancer before requiring a feasible partition.
    OverloadBalancer balancer(ctx);
    balancer.initialize(p_graph);
    arena.execute([&] { balancer.refine(p_graph, ctx.partition); });
    check_weights(p_graph, ctx.partition);
  }
}

class FMConcurrentGainTest : public ::testing::TestWithParam<int> {};

TEST_P(FMConcurrentGainTest, AdjacentSearchesRemainConsistentWithCachedGains) {
  const int num_threads = GetParam();
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, num_threads);
  tbb::task_arena arena(num_threads);
  // Each pair has a gain-1 move in either direction. Moving both endpoints on
  // stale gains creates two new weight-9 cut edges and retains the weight-10 edge.
  constexpr NodeID pairs = 4;
  GraphBuilder builder;
  for (NodeID u = 0; u < 2 * pairs; ++u) {
    builder.new_node();
    builder.new_edge(u ^ 1, 10);
    builder.new_edge(2 * pairs + u % 2, 9);
  }
  for (NodeID side = 0; side < 2; ++side) {
    builder.new_node(1000);
    for (NodeID u = side; u < 2 * pairs; u += 2) {
      builder.new_edge(u, 9);
    }
  }
  const Graph graph = builder.build();
  std::vector<BlockID> partition(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    partition[u] = u % 2;
  }
  Context ctx = create_default_context();
  ctx.parallel.num_threads = num_threads;
  ctx.partition.setup(graph, {1005, 1005});
  ctx.refinement.kway_fm.num_iterations = 1;
  ctx.refinement.kway_fm.num_seed_nodes = 1;
  ctx.refinement.kway_fm.minimal_parallelism = 8;

  for (int repetition = 0; repetition < 1000; ++repetition) {
    SCOPED_TRACE(repetition);
    ctx.refinement.kway_fm.gain_cache_strategy = std::array{
        GainCacheStrategy::COMPACT_HASHING,
        GainCacheStrategy::SPARSE,
        GainCacheStrategy::ON_THE_FLY,
        GainCacheStrategy::AUTO
    }[repetition % 4];
    auto p_graph = make_p_graph(graph, 2, partition);
    FMRefiner refiner(ctx);
    refiner.initialize(p_graph);
    arena.execute([&] { refiner.refine(p_graph, ctx.partition); });
    // Single-threaded gains are exact. Concurrent searches deliberately accept cached
    // gains, so their combined cut is allowed to increase.
    if (num_threads == 1) {
      ASSERT_LE(metrics::edge_cut(p_graph), 40);
    }
    OverloadBalancer balancer(ctx);
    balancer.initialize(p_graph);
    arena.execute([&] { balancer.refine(p_graph, ctx.partition); });
    check_weights(p_graph, ctx.partition);
  }
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, FMConcurrentGainTest, ::testing::Values(1, 8));

TEST(FMCorrectnessTest, ResetsAbortBetweenRefineCalls) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  constexpr NodeID leaves = 128;
  GraphBuilder builder;
  for (NodeID u = 0; u < leaves; ++u) {
    builder.new_node();
    builder.new_edge(leaves);
  }
  builder.new_node(129);
  for (NodeID u = 0; u < leaves; ++u) {
    builder.new_edge(u);
  }
  const Graph graph = builder.build();
  std::vector<BlockID> partition(graph.n(), 0);
  partition.back() = 1;
  auto p_graph = make_p_graph(graph, 2, partition);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {128, 257});
  ctx.refinement.kway_fm.num_iterations = 1;
  ctx.refinement.kway_fm.num_seed_nodes = graph.n();
  ctx.refinement.kway_fm.minimal_parallelism = 1;
  FMRefiner refiner(ctx);
  refiner.initialize(p_graph);
  for (int invocation = 0; invocation < 2; ++invocation) {
    SCOPED_TRACE(invocation);
    for (NodeID u = 0; u < leaves; ++u) {
      p_graph.set_block(u, 0);
    }
    refiner.refine(p_graph, ctx.partition);
    EXPECT_EQ(metrics::edge_cut(p_graph), 0);
    check_weights(p_graph, ctx.partition);
  }
}

TEST(FMCorrectnessTest, CanReuseRefinerWithMoreBlocks) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  constexpr NodeID n = 16;
  EdgeBasedGraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) {
    builder.add_edge(u, (u + 1) % n, 1 + u % 3);
  }
  const Graph graph = builder.build();
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.refinement.kway_fm.num_iterations = 2;
  ctx.refinement.kway_fm.num_seed_nodes = 1;
  ctx.partition.setup(graph, {9, 9});
  std::vector<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    partition[u] = u % 2;
  }
  auto first = make_p_graph(graph, 2, partition);
  FMRefiner refiner(ctx);
  refiner.initialize(first);
  refiner.refine(first, ctx.partition);
  check_weights(first, ctx.partition);

  ctx.partition.setup(graph, {5, 5, 5, 5});
  for (NodeID u = 0; u < n; ++u) {
    partition[u] = u % 4;
  }
  auto second = make_p_graph(graph, 4, partition);
  refiner.refine(second, ctx.partition);
  OverloadBalancer balancer(ctx);
  balancer.initialize(second);
  balancer.refine(second, ctx.partition);
  check_weights(second, ctx.partition);
}

TEST(FMCorrectnessTest, CompletesProductiveSearchesBeyondInitialSeedNodes) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  constexpr NodeID leaves = 128;
  GraphBuilder builder;
  for (NodeID u = 0; u < leaves; ++u) {
    builder.new_node();
    for (NodeID v = 0; v < leaves; ++v) {
      if (u != v) {
        builder.new_edge(v);
      }
    }
    builder.new_edge(leaves, 256);
  }
  builder.new_node(129);
  for (NodeID u = 0; u < leaves; ++u) {
    builder.new_edge(u, 256);
  }
  const Graph graph = builder.build();
  std::vector<BlockID> partition(graph.n(), 0);
  partition.back() = 1;
  auto p_graph = make_p_graph(graph, 2, partition);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {128, 257});
  ctx.refinement.kway_fm.num_iterations = 1;
  ctx.refinement.kway_fm.num_seed_nodes = 1;
  FMRefiner refiner(ctx);
  refiner.initialize(p_graph);
  EXPECT_TRUE(refiner.refine(p_graph, ctx.partition));
  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
  check_weights(p_graph, ctx.partition);
}

TEST(FMCorrectnessTest, AutomaticSeedDefaultsPreserveBudgetAndExplicitChoices) {
  auto fm_ctx = create_eco_context().refinement.kway_fm;
  ASSERT_EQ(fm_ctx.num_seed_nodes, 0u);
  EXPECT_EQ(fm_ctx.num_seed_nodes_for_fm(), 100u);
  EXPECT_EQ(fm_ctx.num_seed_nodes_for_ufm(100, 599), 25u);
  EXPECT_EQ(fm_ctx.num_seed_nodes_for_ufm(100, 600), 75u);
  fm_ctx.num_seed_nodes = 3;
  EXPECT_EQ(fm_ctx.num_seed_nodes_for_fm(), 3u);
  EXPECT_EQ(fm_ctx.num_seed_nodes_for_ufm(100, 599), 3u);
  EXPECT_EQ(fm_ctx.num_seed_nodes_for_ufm(100, 600), 3u);
}

TEST(FMCorrectnessTest, AutomaticSeedCountsMakeProgressInBothRefiners) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  GraphBuilder builder;
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_node();
    builder.new_edge(4);
  }
  builder.new_node(4);
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_edge(u);
  }
  const Graph graph = builder.build();
  Context ctx = create_eco_context();
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {5, 5});
  ctx.refinement.kway_fm.num_iterations = 1;
  ctx.refinement.kway_fm.num_seed_nodes = 0;
  ctx.refinement.kway_fm.unconstrained_upper_bound = 1.0;
  for (const bool unconstrained : {false, true}) {
    auto p_graph = make_p_graph(graph, 2, {0, 0, 0, 0, 1});
    std::unique_ptr<Refiner> refiner;
    if (unconstrained) {
      refiner = std::make_unique<UnconstrainedFMRefiner>(ctx);
    } else {
      refiner = std::make_unique<FMRefiner>(ctx);
    }
    refiner->initialize(p_graph);
    EXPECT_TRUE(refiner->refine(p_graph, ctx.partition));
    EXPECT_EQ(metrics::edge_cut(p_graph), 3);
    check_weights(p_graph, ctx.partition);
  }
}

TEST(FMCorrectnessTest, SparseUFMHonorsConfiguredOverloadLimit) {
#ifndef KAMINPAR_ENABLE_TIMERS
  GTEST_SKIP() << "Requires phase counters to observe speculative overload before rollback";
#else
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  GraphBuilder builder;
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_node();
    builder.new_edge(4);
  }
  builder.new_node(4);
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_edge(u);
  }
  const Graph graph = builder.build();
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {5, 5});
  ctx.refinement.kway_fm.num_iterations = 1;
  ctx.refinement.kway_fm.unconstrained_penalty_min = 0.0;
  ctx.refinement.kway_fm.unconstrained_penalty_max = 0.0;
  ctx.refinement.kway_fm.unconstrained_upper_bound = 1.0;
  auto p_graph = make_p_graph(graph, 2, {0, 0, 0, 0, 1});
  Timer::global().reset();
  UnconstrainedFMRefiner refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);
  const auto &root = Timer::global().tree();
  const auto fm = root.children_tbl.find("FM");
  ASSERT_NE(fm, root.children_tbl.end());
  EXPECT_EQ(fm->second->children_tbl.count("Rebalance"), 0u);
  EXPECT_EQ(metrics::edge_cut(p_graph), 3);
  check_weights(p_graph, ctx.partition);
  Timer::global().reset();
#endif
}

TEST(FMCorrectnessTest, StalledUnconstrainedRoundStillRunsConstrainedFallback) {
#ifndef KAMINPAR_ENABLE_TIMERS
  GTEST_SKIP() << "Requires phase counters to observe the constrained fallback";
#else
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  GraphBuilder builder;
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_node();
    builder.new_edge(4);
  }
  builder.new_node(4);
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_edge(u);
  }
  const Graph graph = builder.build();
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {5, 5});
  ctx.refinement.kway_fm.num_iterations = 4;
  ctx.refinement.kway_fm.unconstrained_num_iterations = 3;
  ctx.refinement.kway_fm.unconstrained_upper_bound = 1.0;
  ctx.refinement.kway_fm.unconstrained_penalty_min = 0.0;
  ctx.refinement.kway_fm.unconstrained_penalty_max = 0.0;
  for (const double threshold : {.005, 0.0, -1.0}) {
    SCOPED_TRACE(threshold);
    ctx.refinement.kway_fm.unconstrained_min_improvement = threshold;
    auto p_graph = make_p_graph(graph, 2, {0, 0, 0, 0, 1});
    Timer::global().reset();
    UnconstrainedFMRefiner refiner(ctx);
    refiner.initialize(p_graph);
    refiner.refine(p_graph, ctx.partition);
    const auto &fm = Timer::global().tree().children_tbl.at("FM");
    // One improving unconstrained round and one stalled round. A nonnegative
    // threshold also requests a constrained attempt; a negative value disables it.
    EXPECT_EQ(
        fm->children_tbl.at("Localized searches, fine level")->restarts, threshold >= 0.0 ? 3u : 2u
    );
    EXPECT_EQ(metrics::edge_cut(p_graph), 3);
    check_weights(p_graph, ctx.partition);
    Timer::global().reset();
  }
#endif
}

} // namespace
} // namespace kaminpar::shm::testing
