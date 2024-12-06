#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/delta_partitioned_graph.h"

namespace kaminpar::shm::testing {

namespace {

DeltaPartitionedGraph make_d_graph(const PartitionedGraph &p_graph) {
  return {&p_graph};
}

} // namespace

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_original_blocks) {
  auto graph = make_empty_graph(2);
  auto p_graph = make_p_graph(graph, 2, {0, 0});
  auto d_graph = make_d_graph(p_graph);

  EXPECT_EQ(d_graph.block(0), p_graph.block(0));
  EXPECT_EQ(d_graph.block(1), p_graph.block(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_new_blocks) {
  auto graph = make_empty_graph(2);
  auto p_graph = make_p_graph(graph, 2, {0, 0});
  auto d_graph = make_d_graph(p_graph);

  EXPECT_EQ(d_graph.block(0), p_graph.block(0));
  d_graph.set_block(0, 1);
  EXPECT_NE(d_graph.block(0), p_graph.block(0));
  EXPECT_EQ(d_graph.block(0), 1);

  EXPECT_EQ(d_graph.block(1), p_graph.block(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_original_block_weights) {
  auto graph = make_empty_graph(2);
  auto p_graph = make_p_graph(graph, 2, {0, 0});
  auto d_graph = make_d_graph(p_graph);

  EXPECT_EQ(d_graph.block_weight(0), p_graph.block_weight(0));
  EXPECT_EQ(d_graph.block_weight(1), p_graph.block_weight(1));
}

TEST(DeltaPartitionedGraphTest, two_node_graph_delta_returns_modified_block_weights) {
  auto graph = make_empty_graph(2);
  auto p_graph = make_p_graph(graph, 2, {0, 0});
  auto d_graph = make_d_graph(p_graph);

  d_graph.set_block(0, 1);
  d_graph.set_block(1, 1);

  EXPECT_EQ(d_graph.block_weight(0), 0);
  EXPECT_EQ(d_graph.block_weight(1), p_graph.total_node_weight());
}

} // namespace kaminpar::shm::testing
