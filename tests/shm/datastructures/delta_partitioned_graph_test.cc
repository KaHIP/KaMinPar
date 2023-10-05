#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"

namespace kaminpar::shm::testing {
namespace graphs {

template <bool allow_read_after_move, bool compact_block_weight_delta>
GenericDeltaPartitionedGraph<allow_read_after_move, compact_block_weight_delta>
d_graph(const PartitionedGraph &p_graph) {
  return GenericDeltaPartitionedGraph<allow_read_after_move, compact_block_weight_delta>(&p_graph);
}
} // namespace graphs

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_original_blocks_ram) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, false>(p_graph);

  EXPECT_EQ(d_graph.block(0), p_graph.block(0));
  EXPECT_EQ(d_graph.block(1), p_graph.block(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_original_blocks_nram) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<false, false>(p_graph);

  EXPECT_EQ(d_graph.block(0), p_graph.block(0));
  EXPECT_EQ(d_graph.block(1), p_graph.block(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_new_blocks_ram) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, false>(p_graph);

  EXPECT_EQ(d_graph.block(0), p_graph.block(0));
  d_graph.set_block(0, 1);
  EXPECT_NE(d_graph.block(0), p_graph.block(0));
  EXPECT_EQ(d_graph.block(0), 1);

  EXPECT_EQ(d_graph.block(1), p_graph.block(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_new_blocks_nram) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, false>(p_graph);

  EXPECT_EQ(d_graph.block(0), p_graph.block(0));
  d_graph.set_block(0, 1);
  // -- d_graph.block(0) query not allowed

  EXPECT_EQ(d_graph.block(1), p_graph.block(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_original_block_weights_compact) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, true>(p_graph);

  EXPECT_EQ(d_graph.block_weight(0), p_graph.block_weight(0));
  EXPECT_EQ(d_graph.block_weight(1), p_graph.block_weight(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_original_block_weights) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, false>(p_graph);

  EXPECT_EQ(d_graph.block_weight(0), p_graph.block_weight(0));
  EXPECT_EQ(d_graph.block_weight(1), p_graph.block_weight(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_modified_block_weights_compact) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, true>(p_graph);

  d_graph.set_block(0, 1);
  d_graph.set_block(1, 1);

  EXPECT_EQ(d_graph.block_weight(0), 0);
  EXPECT_EQ(d_graph.block_weight(1), p_graph.total_node_weight());
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_modified_block_weights) {
  auto graph = graphs::empty(2);
  auto p_graph = graphs::p_graph(graph, 2);
  auto d_graph = graphs::d_graph<true, false>(p_graph);

  d_graph.set_block(0, 1);
  d_graph.set_block(1, 1);

  EXPECT_EQ(d_graph.block_weight(0), 0);
  EXPECT_EQ(d_graph.block_weight(1), p_graph.total_node_weight());
}
} // namespace kaminpar::shm::testing
