#include <limits>
#include <tuple>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "dist/distributed_graph_builder.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/refinement/balancer/node_balancer.h"

namespace kaminpar::dist {

class NodeBalancerLifecycleTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

TEST_P(NodeBalancerLifecycleTest, FeasibleCallsCanPrecedeAndFollowRealBalancing) {
  const auto [parallel_only, update_pq_gains, high_degree_shortcut] = GetParam();
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  constexpr NodeID kNodesPerRank = 8;
  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(kNodesPerRank);
  for (NodeID u = 0; u < kNodesPerRank; ++u) {
    builder.create_node(1);
    for (NodeID v = 0; v < kNodesPerRank; ++v) {
      if (u != v) {
        builder.create_edge(1, rank * kNodesPerRank + v);
      }
    }
  }
  DistributedGraph graph(std::make_unique<DistributedCSRGraph>(builder.finalize()));

  // Parallel-only balancing must use the buckets and the selected temporary-gain branch.
  {
    StaticArray<BlockID> partition(graph.total_n());
    for (const NodeID u : graph.nodes()) {
      partition[u] = u % 2;
    }
    DistributedPartitionedGraph p_graph(&graph, 2, std::move(partition));
    Context ctx = create_default_context();
    ctx.partition.setup(graph.csr_graph(), 2, 0.0);
    ctx.refinement.node_balancer.enable_sequential_balancing = !parallel_only;
    ctx.refinement.node_balancer.enable_parallel_balancing = true;
    ctx.refinement.node_balancer.max_num_rounds = 100;
    ctx.refinement.node_balancer.par_update_pq_gains = update_pq_gains;
    ctx.refinement.node_balancer.par_high_degree_update_thresold =
        high_degree_shortcut ? 0 : std::numeric_limits<EdgeID>::max();
    ctx.refinement.node_balancer.par_high_degree_update_interval = 1;
    NodeBalancerFactory factory(ctx);
    auto balancer = factory.create(p_graph, ctx.partition);

    ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
    balancer->initialize();
    balancer->refine();
    EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));

    // Reuse the same balancer after a real partition change. Replicated weights are
    // rebuilt collectively, as required by the distributed partition contract.
    p_graph.set_block<false>(1, 0);
    p_graph.set_block<false>(3, 0);
    p_graph.reinit_block_weights();
    ASSERT_FALSE(metrics::is_feasible(p_graph, ctx.partition));
    balancer->initialize();
    balancer->refine();
    ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));

    StaticArray<BlockID> balanced_partition(graph.n());
    for (const NodeID u : graph.nodes()) {
      balanced_partition[u] = p_graph.block(u);
    }
    balancer->initialize();
    balancer->refine();
    EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
    for (const NodeID u : graph.nodes()) {
      EXPECT_EQ(p_graph.block(u), balanced_partition[u]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigurations,
    NodeBalancerLifecycleTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool())
);

} // namespace kaminpar::dist
