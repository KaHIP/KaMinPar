/*******************************************************************************
 * Minimum block weight constraints in shared-memory k-way FM refinement.
 ******************************************************************************/
#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/initial_partitioning/refinement/initial_fm_refiner.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/fm/fm_refiner.h"

namespace kaminpar::shm {
namespace {

using namespace testing;

void refine_with_min_weights(
    PartitionedGraph &p_graph,
    std::vector<BlockWeight> min_weights,
    const int num_threads,
    BlockWeight max_weight = 0
) {
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.parallel.num_threads = num_threads;
  if (max_weight == 0) {
    max_weight = p_graph.graph().total_node_weight();
  }
  ctx.partition.setup(p_graph.graph(), std::vector<BlockWeight>(p_graph.k(), max_weight));
  ctx.partition.setup_min_block_weights(std::move(min_weights));
  config.refinement.kway_fm.num_iterations = 4;
  config.refinement.kway_fm.num_seed_nodes = 1;
  config.refinement.kway_fm.minimal_parallelism = num_threads;

  tbb::task_arena arena(num_threads);
  arena.execute([&] {
    FMRefiner refiner(ctx);
    refiner.initialize(p_graph);
    refiner.refine(p_graph, ctx.partition);
  });

  for (const BlockID b : p_graph.blocks()) {
    EXPECT_GE(p_graph.block_weight(b), ctx.partition.min_block_weight(b));
    EXPECT_LE(p_graph.block_weight(b), ctx.partition.max_block_weight(b));
  }
}

Graph make_weighted_star() {
  GraphBuilder builder;
  builder.new_node(3);
  for (NodeID leaf = 1; leaf <= 8; ++leaf) {
    builder.new_edge(leaf);
  }
  for (NodeID leaf = 1; leaf <= 8; ++leaf) {
    builder.new_node(2);
    builder.new_edge(0);
  }
  return builder.build();
}

TEST(FMMinBlockWeightsTest, DoesNotMoveNodesOutOfBlocksAtTheirMinimum) {
  const Graph graph = make_weighted_star();
  auto p_graph = make_p_graph(graph, 2, {0, 1, 1, 1, 1, 1, 1, 1, 1});

  refine_with_min_weights(p_graph, {3, 16}, 1);

  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 16);
  EXPECT_EQ(metrics::edge_cut(p_graph), 8);
}

TEST(FMMinBlockWeightsTest, UsesAvailableSlackWithWeightedNodesAndAsymmetricMinima) {
  const Graph graph = make_weighted_star();
  auto p_graph = make_p_graph(graph, 2, {0, 1, 1, 1, 1, 1, 1, 1, 1});

  refine_with_min_weights(p_graph, {2, 8}, 1);

  EXPECT_LT(metrics::edge_cut(p_graph), 8);
}

TEST(FMMinBlockWeightsTest, ConcurrentBatchesPreserveBothWeightBounds) {
  const Graph graph = make_matching_graph(128);
  std::vector<BlockID> partition(graph.n());
  for (NodeID u = 0; u < graph.n(); ++u) {
    partition[u] = u % 2;
  }

  for (int repetition = 0; repetition < 20; ++repetition) {
    auto p_graph = make_p_graph(graph, 2, partition);
    refine_with_min_weights(p_graph, {127, 127}, 4, 129);
    EXPECT_LT(metrics::edge_cut(p_graph), 128);
  }
}

template <typename Refiner> class InitialFMMinBlockWeightsTest : public ::testing::Test {
public:
  void refine(PartitionedGraph &p_graph, std::vector<BlockWeight> min_weights) {
    Context config = create_default_context();
    kaminpar::RuntimeContext runtime;
    ExecutionContext ctx{config, runtime};
    ctx.partition.setup(
        p_graph.graph(), std::vector<BlockWeight>(p_graph.k(), p_graph.graph().total_node_weight())
    );
    ctx.partition.setup_min_block_weights(std::move(min_weights));
    auto &fm_ctx = config.initial_partitioning.pool.refinement.fm;
    fm_ctx.num_iterations = 4;

    auto csr_view = p_graph.csr_view();
    Refiner refiner(fm_ctx, runtime.random.local());
    refiner.init(csr_view.graph());
    refiner.refine(csr_view, ctx.partition);

    for (const BlockID b : p_graph.blocks()) {
      EXPECT_GE(p_graph.block_weight(b), ctx.partition.min_block_weight(b));
    }
  }
};

using InitialFMTypes = ::testing::Types<InitialSimple2WayFM, InitialAdaptive2WayFM>;
TYPED_TEST_SUITE(InitialFMMinBlockWeightsTest, InitialFMTypes);

TYPED_TEST(InitialFMMinBlockWeightsTest, RollsBackCutsThatViolateMinimumWeights) {
  const Graph graph = make_weighted_star();
  auto p_graph = make_p_graph(graph, 2, {0, 1, 1, 1, 1, 1, 1, 1, 1});

  this->refine(p_graph, {3, 16});

  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 16);
  EXPECT_EQ(metrics::edge_cut(p_graph), 8);
}

TYPED_TEST(InitialFMMinBlockWeightsTest, AcceptsImprovingCutsWithWeightedNodes) {
  // The isolated node retains the minimum weight of block 0 when its boundary node moves.
  const Graph graph = make_graph({0, 1, 2, 2}, {1, 0}, {3, 2, 3}, {1, 1});
  auto p_graph = make_p_graph(graph, 2, {0, 1, 0});

  this->refine(p_graph, {3, 2});

  EXPECT_EQ(p_graph.block_weight(0), 3);
  EXPECT_EQ(p_graph.block_weight(1), 5);
  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
}

TYPED_TEST(InitialFMMinBlockWeightsTest, AllowsSpeculativeMovesBeforeAFeasibleExchange) {
  const Graph graph = make_graph({0, 1, 2, 3, 4}, {1, 0, 3, 2}, {2, 2, 2, 2}, {1, 1, 1, 1});
  auto p_graph = make_p_graph(graph, 2, {0, 1, 0, 1});

  this->refine(p_graph, {4, 4});

  EXPECT_EQ(p_graph.block_weight(0), 4);
  EXPECT_EQ(p_graph.block_weight(1), 4);
  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
}

} // namespace
} // namespace kaminpar::shm
