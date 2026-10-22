/*******************************************************************************
 * Integration tests for the adaptive cache in constrained and unconstrained FM.
 ******************************************************************************/
#include <algorithm>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"
#include "kaminpar-shm/refinement/fm/unconstrained_fm_refiner.h"

namespace kaminpar::shm::testing {
namespace {

// Uniform degrees avoid conflating cache behavior with graph-order assumptions.
// Degree four exercises the stable path; degree eighteen exercises snapshot rollback.
Graph make_ufm_cache_graph(const bool dense, const bool weighted) {
  constexpr NodeID n = 96;
  std::vector<EdgeID> offsets{0};
  std::vector<NodeID> edges;
  std::vector<EdgeWeight> edge_weights;
  std::vector<NodeWeight> node_weights;
  for (NodeID u = 0; u < n; ++u) {
    std::vector<NodeID> neighbors;
    for (NodeID distance = 1; distance <= (dense ? 9u : 2u); ++distance) {
      neighbors.push_back((u + distance) % n);
      neighbors.push_back((u + n - distance) % n);
    }
    std::sort(neighbors.begin(), neighbors.end());
    for (const NodeID v : neighbors) {
      edges.push_back(v);
      edge_weights.push_back(weighted ? 1 + (u + v) % 7 : 1);
    }
    offsets.push_back(edges.size());
    node_weights.push_back(weighted ? 1 + (u / 4) % 3 : 1);
  }
  return make_graph(offsets, edges, node_weights, edge_weights);
}

class UnconstrainedFMJetCacheTest
    : public ::testing::TestWithParam<std::tuple<bool, int, bool, bool>> {};

TEST_P(UnconstrainedFMJetCacheTest, RefinesAndReinitializesWithFeasibleNonincreasingCut) {
  const auto [compressed, threads, dense, weighted] = GetParam();
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
  auto graph = make_ufm_cache_graph(dense, weighted);
  if (compressed) {
    graph = compress(graph.csr_graph());
  }
  constexpr BlockID k = 4;
  std::vector<BlockID> initial(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    initial[u] = u % k;
  }

  for (const GainCacheStrategy strategy :
       {GainCacheStrategy::COMPACT_HASHING, GainCacheStrategy::JET}) {
    SCOPED_TRACE(static_cast<int>(strategy));
    KaMinPar::reseed(42);
    Context ctx = create_eco_context();
    ctx.parallel.num_threads = threads;
    ctx.partition.setup(graph, k, 0.03);
    ctx.refinement.kway_fm.gain_cache_strategy = strategy;
    ctx.refinement.kway_fm.num_seed_nodes = 4;
    ctx.refinement.kway_fm.num_iterations = 4;
    ctx.refinement.kway_fm.unconstrained_num_iterations = 3;
    ctx.refinement.kway_fm.minimal_parallelism = threads;
    auto p_graph = make_p_graph(graph, k, initial);
    ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
    UnconstrainedFMRefiner refiner(ctx);
    refiner.initialize(p_graph);
    for (int round = 0; round < 2; ++round) {
      const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
      refiner.refine(p_graph, ctx.partition);
      EXPECT_LE(metrics::edge_cut(p_graph), initial_cut);
      EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
      std::vector<BlockWeight> actual(k, 0);
      for (NodeID u = 0; u < graph.n(); ++u) {
        ASSERT_LT(p_graph.block(u), k);
        actual[p_graph.block(u)] += p_graph.node_weight(u);
      }
      for (BlockID b = 0; b < k; ++b) {
        EXPECT_EQ(p_graph.block_weight(b), actual[b]);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    RepresentationsThreadsPathsAndWeights,
    UnconstrainedFMJetCacheTest,
    ::testing::Combine(
        ::testing::Bool(), ::testing::Values(1, 4), ::testing::Bool(), ::testing::Bool()
    )
);

class ConstrainedFMJetCacheTest : public ::testing::TestWithParam<std::tuple<bool, int>> {};

TEST_P(ConstrainedFMJetCacheTest, SharedCacheOptionRefinesWeightedGraph) {
  const auto [compressed, threads] = GetParam();
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
  auto graph = make_ufm_cache_graph(false, true);
  if (compressed) {
    graph = compress(graph.csr_graph());
  }
  constexpr BlockID k = 4;
  std::vector<BlockID> initial(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    initial[u] = u % k;
  }

  KaMinPar::reseed(42);
  Context ctx = create_default_context();
  ctx.parallel.num_threads = threads;
  ctx.partition.setup(graph, k, 0.10);
  ctx.refinement.kway_fm.gain_cache_strategy = get_gain_cache_strategies().at("jet");
  ASSERT_EQ(ctx.refinement.kway_fm.gain_cache_strategy, GainCacheStrategy::JET);
  ctx.refinement.kway_fm.num_seed_nodes = 4;
  ctx.refinement.kway_fm.num_iterations = 4;
  ctx.refinement.kway_fm.minimal_parallelism = threads;
  auto p_graph = make_p_graph(graph, k, initial);
  ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  const EdgeWeight initial_cut = metrics::edge_cut(p_graph);

  FMRefiner refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);

  EXPECT_LT(metrics::edge_cut(p_graph), initial_cut);
  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  std::vector<BlockWeight> actual(k, 0);
  for (NodeID u = 0; u < graph.n(); ++u) {
    ASSERT_LT(p_graph.block(u), k);
    actual[p_graph.block(u)] += p_graph.node_weight(u);
  }
  for (BlockID b = 0; b < k; ++b) {
    EXPECT_EQ(p_graph.block_weight(b), actual[b]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    RepresentationsAndThreads,
    ConstrainedFMJetCacheTest,
    ::testing::Combine(::testing::Bool(), ::testing::Values(1, 4))
);

} // namespace
} // namespace kaminpar::shm::testing
