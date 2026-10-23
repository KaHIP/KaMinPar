#include <vector>

#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/multi_queue_overload_balancer.h"
#include "kaminpar-shm/refinement/balancer/overload_balancer.h"
#include "kaminpar-shm/refinement/jet/jet_refiner.h"

namespace kaminpar::shm::testing {
namespace {

TEST(MinBlockWeightsPresetTest, PresetsProducePartitionsWithinConfiguredBounds) {
  const Graph graph = make_grid_graph(16, 16);
  const CSRGraph &csr = graph.csr_graph();
  const std::vector<BlockWeight> minima = {48, 50, 52, 54};
  int preset = 0;
  for (Context ctx : {
           create_eco_context(),
           create_strong_context(),
           create_largek_eco_context(),
           create_largek_strong_context(),
           create_jet_context(2),
       }) {
    SCOPED_TRACE(preset++);
    KaMinPar partitioner(4, std::move(ctx));
    partitioner.set_output_level(OutputLevel::QUIET);
    partitioner.copy_graph(csr.raw_nodes().view(), csr.raw_edges().view());
    partitioner.set_k(4);
    partitioner.set_uniform_max_block_weights(0.25);
    partitioner.set_absolute_min_block_weights(minima);
    std::vector<BlockID> partition(graph.n());
    partitioner.compute_partition(partition);

    std::vector<BlockWeight> weights(4, 0);
    for (const BlockID block : partition) {
      ++weights[block];
    }
    for (BlockID block = 0; block < 4; ++block) {
      EXPECT_GE(weights[block], minima[block]);
      EXPECT_LE(weights[block], 80);
    }
  }
}

TEST(JetMinBlockWeightsTest, RejectsUnderloadingMovesAndUsesAvailableSlack) {
  constexpr NodeID kLeaves = 128;
  tbb::task_arena arena(4);
  for (const bool compressed : {false, true}) {
    const Graph csr = make_star_graph(kLeaves);
    const Graph graph = compressed ? compress(csr.csr_graph()) : make_star_graph(kLeaves);
    std::vector<BlockID> partition(graph.n(), 1);
    partition[0] = 0;
    for (const auto cache : {JetGainCacheStrategy::SPARSE, JetGainCacheStrategy::JET}) {
      for (const auto balancer : {JetBalancingAlgorithm::OVERLOAD, JetBalancingAlgorithm::JET}) {
        Context ctx = create_default_context();
        ctx.parallel.num_threads = 4;
        ctx.refinement.jet.gain_cache_strategy = cache;
        ctx.refinement.jet.balancing_algorithm = balancer;
        ctx.refinement.jet.num_iterations = 2;
        ctx.refinement.jet.num_rounds_on_fine_level = 2;
        ctx.partition.setup(graph, {kLeaves + 1, kLeaves + 1});

        // All leaves want the center's block. The last-slot case makes their
        // source-bound transactions contend for the available donor weight.
        for (const BlockWeight leaf_minimum : {128, 127, 64}) {
          for (int repetition = 0; repetition < 4; ++repetition) {
            SCOPED_TRACE(
                ::testing::Message()
                << "compressed=" << compressed << " cache=" << static_cast<int>(cache)
                << " balancer=" << static_cast<int>(balancer) << " minimum=" << leaf_minimum
                << " repetition=" << repetition
            );
            ctx.partition.setup_min_block_weights({1, leaf_minimum});
            auto p_graph = make_p_graph(graph, 2, partition);
            arena.execute([&] {
              JetRefiner refiner(ctx);
              refiner.initialize(p_graph);
              refiner.refine(p_graph, ctx.partition);
            });

            EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
            EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
            if (leaf_minimum == kLeaves) {
              EXPECT_EQ(p_graph.block(0), 0);
              EXPECT_EQ(p_graph.block_weight(1), kLeaves);
              EXPECT_EQ(metrics::edge_cut(p_graph), kLeaves);
              for (NodeID node = 1; node < graph.n(); ++node) {
                EXPECT_EQ(p_graph.block(node), 1);
              }
            } else {
              EXPECT_LT(metrics::edge_cut(p_graph), kLeaves);
            }
            std::vector<BlockWeight> weights(2, 0);
            for (NodeID node = 0; node < graph.n(); ++node) {
              ASSERT_LT(p_graph.block(node), 2);
              weights[p_graph.block(node)] += p_graph.node_weight(node);
            }
            EXPECT_EQ(p_graph.block_weight(0), weights[0]);
            EXPECT_EQ(p_graph.block_weight(1), weights[1]);
          }
        }
      }
    }
  }
}

template <typename RefinerType> class OverloadMinBlockWeightsTest : public ::testing::Test {};
using OverloadBalancers =
    ::testing::Types<BlockParallelOverloadBalancer, MultiQueueOverloadBalancer>;
TYPED_TEST_SUITE(OverloadMinBlockWeightsTest, OverloadBalancers);

TYPED_TEST(OverloadMinBlockWeightsTest, SkipsHeavyDonorsAndRebalancesWithLighterNodes) {
  GraphBuilder builder;
  builder.new_node(4);
  builder.new_edge(2, 100);
  builder.new_node(2);
  builder.new_edge(2, 1);
  builder.new_node(4);
  builder.new_edge(0, 100);
  builder.new_edge(1, 1);
  const Graph graph = builder.build();
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {4, 10});
  ctx.partition.setup_min_block_weights({3, 1});
  auto p_graph = make_p_graph(graph, 2, {0, 0, 1});

  TypeParam refiner(ctx);
  refiner.initialize(p_graph);
  refiner.refine(p_graph, ctx.partition);

  EXPECT_EQ(p_graph.block(0), 0);
  EXPECT_EQ(p_graph.block(1), 1);
  EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
}

TEST(PartitionedGraphMinBlockWeightsTest, ConcurrentTransfersPreserveBothBoundsAndViews) {
  // Exercise the aligned and dense block weight representations, including zero minima.
  for (const BlockID k : {3, 257}) {
    constexpr NodeID kNodes = 1536;
    const Graph graph = make_empty_graph(kNodes);
    std::vector<BlockID> partition(kNodes);
    for (NodeID u = 0; u < kNodes; ++u) {
      partition[u] = u % 3;
    }
    auto p_graph = make_p_graph(graph, k, partition);
    auto view = p_graph.csr_view();
    const std::vector<BlockWeight> minima = {480, 0, 480};
    tbb::task_arena arena(8);
    for (int iteration = 0; iteration < 8; ++iteration) {
      arena.execute([&] {
        tbb::parallel_for<NodeID>(0, kNodes, [&](const NodeID u) {
          const BlockID from = p_graph.block(u);
          const BlockID to = (from + 1) % 3;
          if (u % 2 == 0) {
            (void)p_graph.move(u, from, to, 544, minima[from]);
          } else {
            (void)view.move(u, from, to, 544, minima[from]);
          }
        });
      });
      std::vector<BlockWeight> weights(k, 0);
      for (NodeID u = 0; u < kNodes; ++u) {
        ++weights[p_graph.block(u)];
      }
      for (const BlockID block : p_graph.blocks()) {
        EXPECT_EQ(p_graph.block_weight(block), weights[block]);
        if (block < 3) {
          EXPECT_GE(weights[block], minima[block]);
          EXPECT_LE(weights[block], 544);
        }
      }
    }
  }
}

} // namespace
} // namespace kaminpar::shm::testing
