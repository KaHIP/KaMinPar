#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <tbb/global_control.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/jet/jet_refiner.h"

namespace kaminpar::shm::testing {
namespace {

Graph make_jet_graph(
    const NodeID n,
    const std::vector<std::pair<NodeID, NodeID>> &edges,
    const EdgeWeight edge_weight = 1,
    const std::vector<NodeWeight> &node_weights = {}
) {
  std::vector<std::vector<NodeID>> neighborhoods(n);
  for (const auto &[u, v] : edges) {
    neighborhoods[u].push_back(v);
    neighborhoods[v].push_back(u);
  }

  std::vector<EdgeID> offsets{0};
  std::vector<NodeID> targets;
  std::vector<EdgeWeight> weights;
  for (auto &neighbors : neighborhoods) {
    std::sort(neighbors.begin(), neighbors.end());
    targets.insert(targets.end(), neighbors.begin(), neighbors.end());
    weights.insert(weights.end(), neighbors.size(), edge_weight);
    offsets.push_back(targets.size());
  }
  return make_graph(offsets, targets, node_weights, weights);
}

class JetRefinerTest : public ::testing::TestWithParam<std::pair<int, bool>> {
protected:
  void SetUp() override {
    _control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, GetParam().first
    );
  }

  Graph graph(
      const NodeID n,
      const std::vector<std::pair<NodeID, NodeID>> &edges,
      const EdgeWeight edge_weight = 1,
      const std::vector<NodeWeight> &node_weights = {}
  ) const {
    Graph result = make_jet_graph(n, edges, edge_weight, node_weights);
    if (GetParam().second) {
      return compress(result.csr_graph());
    }
    return result;
  }

  Context context(const Graph &graph, const BlockID k = 2) const {
    Context ctx = create_default_context();
    ctx.parallel.num_threads = GetParam().first;
    ctx.partition.setup(graph, k, 0.03);
    ctx.refinement.jet.num_iterations = 1;
    ctx.refinement.jet.num_rounds_on_fine_level = 1;
    ctx.refinement.jet.num_rounds_on_coarse_level = 1;
    return ctx;
  }

  static bool refine(const Context &ctx, PartitionedGraph &p_graph) {
    JetRefiner refiner(ctx);
    refiner.initialize(p_graph);
    return refiner.refine(p_graph, ctx.partition);
  }

  std::unique_ptr<tbb::global_control> _control;
};

TEST_P(JetRefinerTest, KeepsZeroGainCandidatesWhenRoundedThresholdIsZero) {
  // Vertices 4 and 5 have zero initial gain and a rounded threshold of zero.
  // Their coordinated moves reduce the cut once vertices 0 and 2 move.
  auto g =
      graph(6, {{0, 1}, {0, 2}, {0, 4}, {1, 2}, {1, 4}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {3, 5}});
  auto p_graph = make_p_graph(g, 2, {1, 0, 0, 1, 1, 0});
  Context ctx = context(g);
  ASSERT_EQ(metrics::edge_cut(p_graph), 6);

  EXPECT_TRUE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 4);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 3);
}

TEST_P(JetRefinerTest, KeepsZeroProjectedGainAfterburnerMoves) {
  // Every first-filter gain is positive. Vertices 1 and 2 have zero
  // projected gain, but are needed for the four-vertex coordinated move.
  auto g = graph(6, {{0, 1}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {1, 3}, {1, 4}, {2, 5}, {3, 4}}, 7);
  auto p_graph = make_p_graph(g, 2, {1, 0, 1, 0, 1, 0});
  Context ctx = context(g);
  ASSERT_EQ(metrics::edge_cut(p_graph), 49);

  EXPECT_TRUE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 28);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 3);
}

TEST_P(JetRefinerTest, KeepsBalancedRepairWhenInitialInfeasiblePartitionHasSmallerCut) {
  auto g = graph(4, {{0, 1}, {0, 2}, {1, 2}});
  auto p_graph = make_p_graph(g, 2, {0, 0, 0, 1});
  Context ctx = context(g);
  ASSERT_EQ(metrics::edge_cut(p_graph), 0);
  ASSERT_FALSE(metrics::is_feasible(p_graph, ctx.partition));

  EXPECT_TRUE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 2);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
}

TEST_P(JetRefinerTest, KeepsBalancedRepairWhenCombinedViolationExceedsBlockWeightRange) {
  // The total weight (6 * weight) fits, but the combined overload and underload
  // (8 * weight) exceeds the signed range in both 32-bit and 64-bit builds.
  constexpr NodeWeight weight = std::numeric_limits<NodeWeight>::max() / 7;
  constexpr BlockWeight target_weight = 2 * weight;
  auto g = graph(6, {}, 1, std::vector<NodeWeight>(6, weight));
  auto p_graph = make_p_graph(g, 3, {0, 0, 0, 0, 0, 0});
  Context ctx = context(g, 3);
  ctx.partition.setup(g, std::vector<BlockWeight>(3, target_weight));
  ctx.partition.setup_min_block_weights(std::vector<BlockWeight>(3, target_weight));
  ASSERT_EQ(metrics::edge_cut(p_graph), 0);
  ASSERT_FALSE(metrics::is_feasible(p_graph, ctx.partition));

  EXPECT_TRUE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  for (const BlockID b : p_graph.blocks()) {
    EXPECT_EQ(p_graph.block_weight(b), target_weight);
  }
}

TEST_P(JetRefinerTest, PreservesFeasiblePartitionWhenNoImprovementExists) {
  auto g = graph(4, {{0, 1}, {2, 3}});
  auto p_graph = make_p_graph(g, 2, {0, 0, 1, 1});
  Context ctx = context(g);

  EXPECT_FALSE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_THAT(p_graph.raw_partition(), ::testing::ElementsAre(0, 0, 1, 1));
}

TEST_P(JetRefinerTest, DoesNotReplaceBalancedSnapshotWithSmallerInfeasibleCut) {
  // The two accepted moves leave weights 5/7. Neither weight 4 nor weight 3
  // fits in the other block's single free slot, so greedy repair cannot finish.
  auto g = graph(4, {{0, 3}, {1, 2}}, 1, {4, 3, 2, 3});
  auto p_graph = make_p_graph(g, 2, {0, 1, 0, 1});
  Context ctx = context(g);
  ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));

  EXPECT_FALSE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 2);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_THAT(p_graph.raw_partition(), ::testing::ElementsAre(0, 1, 0, 1));
}

TEST_P(JetRefinerTest, HandlesUnsortedLowDegreeGraphWithManyParts) {
  constexpr BlockID k = 64;
  constexpr NodeID n = 2 * k;
  std::vector<std::pair<NodeID, NodeID>> edges;
  std::vector<BlockID> partition(n);
  for (BlockID b = 0; b < k; ++b) {
    edges.emplace_back(2 * b, 2 * b + 1);
    partition[2 * b] = b;
    partition[2 * b + 1] = (b + 1) % k;
  }
  auto g = graph(n, edges);
  auto p_graph = make_p_graph(g, k, partition);
  Context ctx = context(g, k);
  ASSERT_FALSE(g.sorted());
  ASSERT_EQ(metrics::edge_cut(p_graph), k);

  EXPECT_TRUE(refine(ctx, p_graph));

  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
}

TEST_P(JetRefinerTest, PreservesFullEdgeWeightRangeWithManyParts) {
  // Packing six block-ID bits into a gain entry would overflow this otherwise
  // valid weight. Only one edge is cut, so the total cut still fits.
  constexpr BlockID k = 64;
  constexpr NodeID n = 2 * k;
  constexpr EdgeWeight weight = static_cast<EdgeWeight>(1)
                                << (std::numeric_limits<EdgeWeight>::digits - 5);
  auto g = graph(n, {{0, 3}}, weight);
  std::vector<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    partition[u] = u / 2;
  }
  auto p_graph = make_p_graph(g, k, partition);
  Context ctx = context(g, k);
  ASSERT_EQ(metrics::edge_cut(p_graph), weight);

  refine(ctx, p_graph);

  EXPECT_LE(metrics::edge_cut(p_graph), weight);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
}

TEST_P(JetRefinerTest, PreservesFeasibilityAndBestCutAcrossMultipleRounds) {
  std::mt19937 random(42);
  constexpr NodeID n = 24;
  constexpr BlockID k = 4;
  for (int instance = 0; instance < 8; ++instance) {
    SCOPED_TRACE(instance);
    std::vector<std::pair<NodeID, NodeID>> edges;
    for (NodeID u = 0; u < n; ++u) {
      for (NodeID v = u + 1; v < n; ++v) {
        if (random() % 5 == 0) {
          edges.emplace_back(u, v);
        }
      }
    }
    auto g = graph(n, edges, 1 + instance);
    std::vector<BlockID> partition(n);
    for (NodeID u = 0; u < n; ++u) {
      partition[u] = u % k;
    }
    auto p_graph = make_p_graph(g, k, partition);
    Context ctx = context(g, k);
    ctx.refinement.jet.num_iterations = 8;
    ctx.refinement.jet.num_rounds_on_fine_level = 4;
    ctx.refinement.jet.initial_gain_temp_on_fine_level = 0.75;
    ctx.refinement.jet.final_gain_temp_on_fine_level = 0.25;
    const EdgeWeight initial_cut = metrics::edge_cut(p_graph);

    const bool improved = refine(ctx, p_graph);

    EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
    EXPECT_LE(metrics::edge_cut(p_graph), initial_cut);
    EXPECT_EQ(improved, metrics::edge_cut(p_graph) < initial_cut);
    std::vector<BlockWeight> actual_weights(k);
    for (NodeID u = 0; u < n; ++u) {
      ASSERT_LT(p_graph.block(u), k);
      actual_weights[p_graph.block(u)] += p_graph.node_weight(u);
    }
    for (BlockID b = 0; b < k; ++b) {
      EXPECT_EQ(p_graph.block_weight(b), actual_weights[b]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CsrAndCompressed,
    JetRefinerTest,
    ::testing::Values(
        std::pair{1, false}, std::pair{4, false}, std::pair{1, true}, std::pair{4, true}
    ),
    [](const auto &info) {
      return std::string(info.param.second ? "Compressed" : "Csr") +
             std::to_string(info.param.first) + "Threads";
    }
);

} // namespace
} // namespace kaminpar::shm::testing

namespace kaminpar::shm {
namespace {
using namespace testing;

void refine_and_check(Graph &graph, const int threads) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
  for (const auto cache :
       {GainCacheStrategy::SPARSE, GainCacheStrategy::JET, GainCacheStrategy::ON_THE_FLY}) {
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

template <bool exact_gains, bool source_block> struct CountingConnections {
  static constexpr bool kIteratesExactGains = exact_gains;
  static constexpr bool kIteratesSourceBlock = source_block;

  EdgeWeight conn(const NodeID u, const BlockID to) const {
    return connections[u][to];
  }

  template <typename Lambda> void gains(const NodeID u, const BlockID from, Lambda &&lambda) {
    ++queries[u];
    for (BlockID to = 0; to < connections[u].size(); ++to) {
      if (source_block || to != from) {
        lambda(to, [&] { return connections[u][to] - (exact_gains ? conn(u, from) : 0); });
      }
    }
  }

  std::vector<std::vector<EdgeWeight>> connections;
  std::vector<std::size_t> queries;
};

template <bool exact_gains, bool source_block> void check_destination_semantics() {
  CountingConnections<exact_gains, source_block> cache{{{10, 8, 8}}, {0}};
  jet::DestinationCache destinations(cache, 1);
  const auto first = destinations.get(0, 0);
  EXPECT_EQ(first.source_connectivity, 10);
  EXPECT_EQ(first.target_connectivity, 8);
  EXPECT_EQ(first.target, 1);
  EXPECT_EQ(first.gain(), -2);
  EXPECT_FALSE(first.accepts(0.1));
  EXPECT_TRUE(destinations.get(0, 0).accepts(0.5));
  EXPECT_EQ(cache.queries[0], 1); // Temperature changes reuse raw connectivity.

  // Moving the vertex changes the source even if no neighboring vertex moved.
  const auto moved = destinations.get(0, 1);
  EXPECT_EQ(moved.source_connectivity, 8);
  EXPECT_EQ(moved.target, 0);
  EXPECT_EQ(moved.gain(), 2);
  EXPECT_EQ(cache.queries[0], 2);
}

TEST(JetDestinationCacheTest, HandlesSourceIterationExactGainsAndTemperatureChanges) {
  check_destination_semantics<false, false>();
  check_destination_semantics<false, true>();
  check_destination_semantics<true, false>();
  check_destination_semantics<true, true>();
}

TEST(JetDestinationCacheTest, InvalidatesMovedVerticesAndNeighborsAndClearsOnRollback) {
  const auto graph = make_graph({0, 1, 3, 4, 4}, {1, 0, 2, 1});
  CountingConnections<false, false> cache{
      {{3, 2, 1}, {3, 2, 1}, {3, 2, 1}, {3, 2, 1}}, {0, 0, 0, 0}
  };
  jet::DestinationCache destinations(cache, graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    EXPECT_EQ(destinations.get(u, 0).target, 1);
    cache.connections[u][2] = 5;
  }

  destinations.invalidate_neighborhood(graph.csr_graph(), 1);
  for (NodeID u = 0; u < 3; ++u) {
    EXPECT_EQ(destinations.get(u, 0).target, 2);
    EXPECT_EQ(cache.queries[u], 2);
  }
  EXPECT_EQ(destinations.get(3, 0).target, 1);
  EXPECT_EQ(cache.queries[3], 1); // Unaffected destinations remain memoized.

  destinations.invalidate_all(graph.csr_graph());
  for (NodeID u = 0; u < graph.n(); ++u) {
    cache.connections[u][1] = 7;
    EXPECT_EQ(destinations.get(u, 0).target, 1);
  }
  EXPECT_EQ(cache.queries, (std::vector<std::size_t>{3, 3, 3, 2}));
}

TEST(JetComponentTest, RejectsUnsupportedSharedGainCacheStrategy) {
  const auto graph = make_path_graph(4);
  auto p_graph = make_p_graph(graph, 2, {0, 0, 1, 1});
  auto ctx = create_jet_context(1);
  ctx.refinement.jet.gain_cache_strategy = GainCacheStrategy::DENSE;
  ctx.partition.setup(graph, 2, 0.03);
  JetRefiner refiner(ctx);
  refiner.initialize(p_graph);
  EXPECT_THROW(refiner.refine(p_graph, ctx.partition), std::invalid_argument);
}

TEST(JetComponentTest, StoredCachesMatchOnTheFlyAcrossTemperatureChangesOnSortedGraph) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
  constexpr NodeID n = 64;
  constexpr BlockID k = 4;
  std::vector<EdgeID> offsets(n + 1);
  std::vector<NodeID> edges;
  std::vector<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    offsets[u] = edges.size();
    for (const NodeID delta : {NodeID{1}, n - 1, NodeID{8}, n - 8}) {
      edges.push_back((u + delta) % n);
    }
    partition[u] = u % k;
  }
  offsets[n] = edges.size();
  auto graph = make_graph(offsets, edges, true);
  ASSERT_TRUE(graph.sorted());
  std::mt19937 random(42);
  std::shuffle(partition.begin(), partition.end(), random);

  for (const auto balancer : {JetBalancingAlgorithm::JET, JetBalancingAlgorithm::OVERLOAD}) {
    std::vector<BlockID> reference;
    for (const auto cache :
         {GainCacheStrategy::SPARSE, GainCacheStrategy::JET, GainCacheStrategy::ON_THE_FLY}) {
      auto p_graph = make_p_graph(graph, k, partition);
      auto ctx = create_jet_context(4);
      ctx.refinement.jet.gain_cache_strategy = cache;
      ctx.refinement.jet.balancing_algorithm = balancer;
      ctx.refinement.jet.num_iterations = 10;
      ctx.refinement.jet.initial_gain_temp_on_fine_level = 0.1;
      ctx.refinement.jet.final_gain_temp_on_fine_level = 0.9;
      ctx.partition.setup(graph, k, 0.1);
      KaMinPar::reseed(42);
      JetRefiner refiner(ctx);
      refiner.initialize(p_graph);
      refiner.refine(p_graph, ctx.partition);
      EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
      const std::vector<BlockID> result(
          p_graph.raw_partition().begin(), p_graph.raw_partition().end()
      );
      if (reference.empty()) {
        reference = result;
      } else {
        EXPECT_EQ(result, reference);
      }
    }
  }
}

TEST(JetComponentTest, ReusesCachesAcrossShrinkingAndGrowingGraphsAndBlockCounts) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
  for (const bool compressed : {false, true}) {
    for (const auto cache :
         {GainCacheStrategy::SPARSE, GainCacheStrategy::JET, GainCacheStrategy::ON_THE_FLY}) {
      auto ctx = create_jet_context(2);
      ctx.refinement.jet.gain_cache_strategy = cache;
      ctx.refinement.jet.num_iterations = 3;
      JetRefiner retained(ctx);
      for (const auto &[n, k] : std::vector<std::pair<NodeID, BlockID>>{
               {32, 4}, {128, 64}, {16, 2}, {96, 48}, {64, 8}
           }) {
        SCOPED_TRACE(n);
        SCOPED_TRACE(k);
        std::vector<EdgeID> offsets(n + 1);
        std::vector<NodeID> edges;
        std::vector<BlockID> partition(n);
        for (NodeID u = 0; u < n; ++u) {
          offsets[u] = edges.size();
          for (NodeID v = 0; v < n; ++v) {
            if (u != v) {
              edges.push_back(v);
            }
          }
          partition[u] = u % k;
        }
        offsets[n] = edges.size();
        auto graph = make_graph(offsets, edges, true);
        if (compressed) {
          graph = compress(graph.csr_graph());
        }
        ASSERT_TRUE(graph.sorted());
        ctx.partition.setup(graph, k, 0.1);
        auto p_graph = make_p_graph(graph, k, partition);
        KaMinPar::reseed(42);
        retained.initialize(p_graph);
        retained.refine(p_graph, ctx.partition);

        auto reference = make_p_graph(graph, k, partition);
        JetRefiner fresh(ctx);
        KaMinPar::reseed(42);
        fresh.initialize(reference);
        fresh.refine(reference, ctx.partition);
        EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
        for (NodeID u = 0; u < graph.n(); ++u) {
          EXPECT_EQ(p_graph.block(u), reference.block(u));
        }
      }
    }
  }
}

TEST(JetComponentTest, ReusesSparseCacheWhenPackedDegreeBucketsAndBlockCountsChange) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
  struct Instance {
    NodeID n;
    BlockID k;
    NodeID radius;
  };
  for (const bool compressed : {false, true}) {
    auto ctx = create_jet_context(2);
    ctx.refinement.jet.gain_cache_strategy = GainCacheStrategy::SPARSE;
    ctx.refinement.jet.num_iterations = 4;
    ctx.refinement.kway_fm.k_based_high_degree_threshold = 1;
    ctx.refinement.kway_fm.constant_high_degree_threshold = 0;
    JetRefiner retained(ctx);
    for (const auto &[n, k, radius] :
         std::vector<Instance>{{64, 16, 1}, {96, 24, 3}, {32, 8, 1}, {128, 32, 2}, {48, 12, 1}}) {
      SCOPED_TRACE(n);
      SCOPED_TRACE(k);
      SCOPED_TRACE(radius);
      std::vector<EdgeID> offsets(n + 1);
      std::vector<NodeID> edges;
      std::vector<BlockID> partition(n);
      for (NodeID u = 0; u < n; ++u) {
        offsets[u] = edges.size();
        for (NodeID delta = 1; delta <= radius; ++delta) {
          edges.push_back((u + delta) % n);
          edges.push_back((u + n - delta) % n);
        }
        partition[u] = u % k;
      }
      offsets[n] = edges.size();
      auto graph = make_graph(offsets, edges, true);
      if (compressed) {
        graph = compress(graph.csr_graph());
      }
      ASSERT_TRUE(graph.sorted());
      ASSERT_LT(2 * radius, k); // Every row uses the packed hash-table path.
      ctx.partition.setup(graph, k, 0.1);
      auto p_graph = make_p_graph(graph, k, partition);
      KaMinPar::reseed(42);
      retained.initialize(p_graph);
      retained.refine(p_graph, ctx.partition);

      auto reference = make_p_graph(graph, k, partition);
      JetRefiner fresh(ctx);
      KaMinPar::reseed(42);
      fresh.initialize(reference);
      fresh.refine(reference, ctx.partition);
      EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
      for (NodeID u = 0; u < graph.n(); ++u) {
        EXPECT_EQ(p_graph.block(u), reference.block(u));
      }
    }
  }
}

TEST(JetComponentTest, JetPresetsSelectAdaptiveCacheAndPaperBalancer) {
  for (const int rounds : {1, 4}) {
    const auto ctx = create_jet_context(rounds);
    EXPECT_EQ(ctx.refinement.jet.gain_cache_strategy, GainCacheStrategy::JET);
    EXPECT_EQ(ctx.refinement.jet.balancing_algorithm, JetBalancingAlgorithm::JET);
    EXPECT_EQ(ctx.refinement.jet.num_rounds_on_fine_level, rounds);
    EXPECT_EQ(ctx.refinement.jet.num_rounds_on_coarse_level, rounds);
  }
}

TEST(JetComponentTest, SparseOptionSupportsFullWeightsInSortedGraph) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
  constexpr BlockID k = 64;
  constexpr NodeID n = 2 * k;
  constexpr EdgeWeight weight = static_cast<EdgeWeight>(1)
                                << (std::numeric_limits<EdgeWeight>::digits - 5);
  std::vector<EdgeID> offsets(n + 1);
  std::vector<NodeID> edges(n);
  std::vector<EdgeWeight> weights(n, weight);
  std::vector<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    offsets[u] = u;
    edges[u] = u ^ 1;
    partition[u] = u / 2;
  }
  offsets[n] = n;
  // Two heavy edges are cut; all degrees are one, hence degree-sorted.
  std::swap(partition[1], partition[2]);
  std::fill(weights.begin() + 4, weights.end(), 1);
  auto graph = make_graph(offsets, edges, {}, weights, true);
  auto ctx = create_jet_context(2);
  ctx.refinement.jet.gain_cache_strategy = GainCacheStrategy::SPARSE;
  ctx.partition.setup(graph, k, 0.03);
  auto p_graph = make_p_graph(graph, k, partition);
  const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
  JetRefiner refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_LE(metrics::edge_cut(p_graph), initial_cut);
}

TEST(JetComponentTest, SparseOptionRefinesSortedGraphWithBothBalancers) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
  constexpr BlockID k = 64;
  constexpr NodeID n = 2 * k;
  std::vector<EdgeID> offsets(n + 1);
  std::vector<NodeID> edges(n);
  std::vector<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    offsets[u] = u;
    edges[u] = u ^ 1;
    partition[u] = (u / 2 + u % 2) % k;
  }
  offsets[n] = n;
  auto graph = make_graph(offsets, edges, true);
  for (const auto balancer : {JetBalancingAlgorithm::OVERLOAD, JetBalancingAlgorithm::JET}) {
    auto ctx = create_jet_context(2);
    ctx.refinement.jet.gain_cache_strategy = GainCacheStrategy::SPARSE;
    ctx.refinement.jet.balancing_algorithm = balancer;
    ctx.partition.setup(graph, k, 0.03);
    auto p_graph = make_p_graph(graph, k, partition);
    JetRefiner refiner(ctx);
    refiner.initialize(p_graph);
    EXPECT_TRUE(refiner.refine(p_graph, ctx.partition));
    EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
    EXPECT_EQ(metrics::edge_cut(p_graph), 0);
  }
}

TEST(JetComponentTest, AllCombinationsKeepFeasibleBestOnCSR) {
  auto graph = make_grid_graph(16, 16);
  refine_and_check(graph, 1);
  refine_and_check(graph, 4);
}

TEST(JetComponentTest, AllCombinationsKeepFeasibleBestOnCompressedGraph) {
  auto csr = make_grid_graph(16, 16);
  auto graph = compress(csr.csr_graph());
  refine_and_check(graph, 4);
}

TEST(JetComponentTest, AdaptiveCachePreservesLegacyMovesWithOneThread) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 1);
  auto graph = make_grid_graph(12, 12);
  std::vector<BlockID> partition(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    partition[u] = u % 8;
  }
  std::vector<BlockID> reference;
  for (const auto cache :
       {GainCacheStrategy::SPARSE, GainCacheStrategy::JET, GainCacheStrategy::ON_THE_FLY}) {
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

TEST(JetComponentTest, DoesNotRollbackToInfeasibleLowerCutInput) {
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

TEST(JetComponentTest, RetainsBalanceProgressWhenWeightedPackingIsImpossible) {
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

TEST(JetComponentTest, PreservesConfiguredMinimumWeights) {
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
