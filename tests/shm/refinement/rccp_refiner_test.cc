#include <gmock/gmock.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/presets.h"
#include "kaminpar-shm/refinement/rccp/rccp_refiner.h"

namespace kaminpar::shm::testing {

TEST(RccpRefinerTest, FindsRepairCertifiedSingletonRotation) {
  EdgeBasedGraphBuilder builder;
  builder.add_edge(0, 3, 10);
  builder.add_edge(2, 5, 5);
  builder.add_edge(4, 1, 1);
  const Graph graph = builder.build();

  PartitionedGraph p_graph = make_p_graph(graph, 3, {0, 0, 1, 1, 2, 2});

  Context ctx = create_default_context();
  ctx.partition.setup(graph, 3, 0.0, false);
  ctx.refinement.rccp.num_iterations = 1;
  ctx.refinement.rccp.enable_mincut_packets = false;
  ctx.refinement.rccp.max_total_packets = 32;
  ctx.refinement.rccp.master_depth = 6;
  ctx.refinement.rccp.master_beam_width = 64;
  ctx.refinement.rccp.master_branching_factor = 16;

  EXPECT_EQ(metrics::edge_cut(p_graph), 16);

  RccpRefiner refiner(ctx);
  refiner.initialize(p_graph);
  EXPECT_TRUE(refiner.refine(p_graph, ctx.partition));

  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
}

TEST(RccpRefinerTest, DoesNotApplyNonimprovingSwaps) {
  EdgeBasedGraphBuilder builder;
  builder.add_edge(0, 1, 1);
  const Graph graph = builder.build();

  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 1});

  Context ctx = create_default_context();
  ctx.partition.setup(graph, 2, 0.0, false);
  ctx.refinement.rccp.num_iterations = 1;
  ctx.refinement.rccp.enable_mincut_packets = false;
  ctx.refinement.rccp.max_total_packets = 8;
  ctx.refinement.rccp.master_depth = 2;
  ctx.refinement.rccp.master_beam_width = 8;
  ctx.refinement.rccp.master_branching_factor = 8;

  RccpRefiner refiner(ctx);
  refiner.initialize(p_graph);
  EXPECT_FALSE(refiner.refine(p_graph, ctx.partition));

  EXPECT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  EXPECT_EQ(metrics::edge_cut(p_graph), 1);
}

} // namespace kaminpar::shm::testing
