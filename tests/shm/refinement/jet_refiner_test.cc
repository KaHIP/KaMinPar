#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <tbb/global_control.h>

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
