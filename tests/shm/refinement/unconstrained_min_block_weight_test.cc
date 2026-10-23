#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/fm/unconstrained_fm_refiner.h"
#include "kaminpar-shm/refinement/lp/unconstrained_lp_refiner.h"

namespace kaminpar::shm::testing {
namespace {

template <typename RefinerType> class UnconstrainedMinBlockWeightTest : public ::testing::Test {};

using UnconstrainedRefiners =
    ::testing::Types<UnconstrainedLabelPropagationRefiner, UnconstrainedFMRefiner>;
TYPED_TEST_SUITE(UnconstrainedMinBlockWeightTest, UnconstrainedRefiners);

TYPED_TEST(UnconstrainedMinBlockWeightTest, RejectsAnImprovingMoveThatWouldUnderloadItsSource) {
  const Graph graph = make_graph({0, 1, 2}, {1, 0}, {3, 7}, {10, 10});
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 1});

  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {10, 10});
  ctx.partition.setup_min_block_weights({3, 2});
  TypeParam refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);

  EXPECT_EQ(p_graph.block(0), 0);
  EXPECT_EQ(p_graph.block(1), 1);
  EXPECT_EQ(metrics::edge_cut(p_graph), 10);
  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
}

TYPED_TEST(UnconstrainedMinBlockWeightTest, AcceptsAnImprovingMoveDownToAnAsymmetricMinimum) {
  const Graph graph = make_graph({0, 1, 3, 4}, {1, 0, 2, 1}, {3, 2, 5}, {1, 1, 10, 10});
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 0, 1});

  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, {10, 10});
  ctx.partition.setup_min_block_weights({3, 1});
  TypeParam refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);

  EXPECT_EQ(p_graph.block(0), 0);
  EXPECT_EQ(p_graph.block(1), 1);
  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(metrics::edge_cut(p_graph), 1);
  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
}

TEST(UnconstrainedLPMinBlockWeightTest, PreservesMinimaWithConcurrentPositiveGainMoves) {
  constexpr NodeID kLeaves = 128;
  GraphBuilder builder;
  for (NodeID u = 0; u < kLeaves; ++u) {
    builder.new_node();
    builder.new_edge(kLeaves);
  }
  builder.new_node(kLeaves);
  for (NodeID u = 0; u < kLeaves; ++u) {
    builder.new_edge(u);
  }
  Graph graph = builder.build();
  std::vector<BlockID> partition(graph.n(), 0);
  partition.back() = 1;

  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.parallel.num_threads = 32;
  ctx.partition.setup(graph, {256, 256});
  ctx.partition.setup_min_block_weights({64, 1});
  for (int repetition = 0; repetition < 8; ++repetition) {
    PartitionedGraph p_graph = make_p_graph(graph, 2, partition);
    UnconstrainedLabelPropagationRefiner refiner(ctx);
    refiner.initialize(p_graph);
    tbb::task_arena arena(32);
    arena.execute([&] { refiner.refine(p_graph, ctx.partition); });

    EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
    EXPECT_LE(metrics::edge_cut(p_graph), kLeaves);
    EXPECT_EQ(p_graph.block_weight(0) + p_graph.block_weight(1), graph.total_node_weight());
  }
}

class UnconstrainedFMMinBlockWeightRollbackTest : public ::testing::TestWithParam<NodeID> {};

TEST_P(UnconstrainedFMMinBlockWeightRollbackTest, KeepsOnlyFeasiblePrefixes) {
  const NodeID n = GetParam();
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) {
    builder.new_node(u == 0 ? 3 : u == 1 ? 2 : 1);
    for (NodeID v = 0; v < n; ++v) {
      if (u != v) {
        builder.new_edge(v);
      }
    }
  }
  Graph graph = builder.build();
  std::vector<BlockID> partition(n, 1);
  partition[0] = partition[1] = 0;

  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {graph.total_node_weight(), graph.total_node_weight()});
  ctx.partition.setup_min_block_weights({3, 2});
  config.refinement.kway_fm.num_iterations = 3;
  config.refinement.kway_fm.unconstrained_num_iterations = 1;

  for (const auto strategy :
       {GainCacheStrategy::COMPACT_HASHING, GainCacheStrategy::SPARSE, GainCacheStrategy::AUTO}) {
    config.refinement.kway_fm.gain_cache_strategy = strategy;
    PartitionedGraph p_graph = make_p_graph(graph, 2, partition);
    const EdgeWeight initial_cut = metrics::edge_cut(p_graph);
    UnconstrainedFMRefiner refiner(ctx);
    refiner.initialize(p_graph);
    refiner.refine(p_graph, ctx.partition);

    EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
    EXPECT_LE(metrics::edge_cut(p_graph), initial_cut);
    EXPECT_EQ(p_graph.block_weight(0) + p_graph.block_weight(1), graph.total_node_weight());
  }
}

// Clique densities exercise sparse snapshot, incremental, and dense snapshot rollback paths.
INSTANTIATE_TEST_SUITE_P(
    RollbackPaths, UnconstrainedFMMinBlockWeightRollbackTest, ::testing::Values(5, 12, 20)
);

} // namespace
} // namespace kaminpar::shm::testing
