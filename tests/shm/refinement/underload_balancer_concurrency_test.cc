#include <vector>

#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/underload_balancer.h"

namespace kaminpar::shm::testing {
namespace {

TEST(UnderloadBalancerConcurrencyTest, ParallelWorkersFillTheSameUnderloadedBlock) {
  // Every donor has edges to the underloaded block. Four workers compete for the final slots
  // and observe the underloaded flag as the target reaches its minimum weight.
  constexpr NodeID kNumNodes = 128;
  const Graph graph = make_complete_graph(kNumNodes);
  Context config = create_default_context();
  kaminpar::RuntimeContext runtime;
  ExecutionContext ctx{config, runtime};
  ctx.parallel.num_threads = 4;
  ctx.partition.setup(graph, {64, 64});
  ctx.partition.setup_min_block_weights({64, 64});

  tbb::task_arena arena(4);
  for (int repetition = 0; repetition < 4; ++repetition) {
    std::vector<BlockID> partition(kNumNodes, 0);
    std::fill(partition.end() - 8, partition.end(), 1);
    auto p_graph = make_p_graph(graph, 2, partition);
    UnderloadBalancer balancer(ctx);
    arena.execute([&] {
      balancer.initialize(p_graph);
      EXPECT_TRUE(balancer.refine(p_graph, ctx.partition));
    });

    std::vector<BlockWeight> actual_weights(2, 0);
    for (NodeID node = 0; node < kNumNodes; ++node) {
      const BlockID block = p_graph.block(node);
      ASSERT_LT(block, actual_weights.size());
      actual_weights[block] += graph.csr_graph().node_weight(node);
    }
    EXPECT_EQ(actual_weights, std::vector<BlockWeight>({64, 64}));
    EXPECT_EQ(p_graph.block_weight(0), actual_weights[0]);
    EXPECT_EQ(p_graph.block_weight(1), actual_weights[1]);
    EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
    EXPECT_TRUE(metrics::is_min_balanced(p_graph, ctx.partition));
  }
}

} // namespace
} // namespace kaminpar::shm::testing
