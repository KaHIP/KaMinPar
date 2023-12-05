/*******************************************************************************
 * @file:   graph_extraction_test.cc
 * @author: Daniel Seemaier
 * @date:   29.04.2022
 * @brief:  Unit tests to test the extraction of block induced subgraphs.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dist/distributed_graph_factories.h"
#include "tests/dist/distributed_graph_helpers.h"

#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/graphutils/subgraph_extractor.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/datastructures/static_array.h"

using testing::ElementsAre;
using testing::UnorderedElementsAre;

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

////////////////////////////////////////////////////////////////////////////////
/// Extract local subgraphs
////////////////////////////////////////////////////////////////////////////////

inline auto extract_local_subgraphs(const DistributedPartitionedGraph &p_graph) {
  return graph::extract_local_block_induced_subgraphs(p_graph);
}

TEST(LocalGraphExtractionTest, extract_local_nodes_from_isolated_nodes_graph_1) {
  auto graph = make_isolated_nodes_graph(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto result = extract_local_subgraphs(p_graph);

  ASSERT_EQ(result.shared_nodes.size(), 1);
  EXPECT_EQ(result.shared_nodes[0], 0);
  EXPECT_EQ(result.shared_edges.size(), 0);
  ASSERT_EQ(result.shared_node_weights.size(), 1);
  EXPECT_EQ(result.shared_node_weights[0], 1);
  EXPECT_EQ(result.shared_edge_weights.size(), 0);
}

TEST(LocalGraphExtractionTest, extract_local_nodes_from_isolated_nodes_graph_2) {
  auto graph = make_isolated_nodes_graph(2);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto result = extract_local_subgraphs(p_graph);

  ASSERT_EQ(result.shared_nodes.size(), 2);
  EXPECT_THAT(result.shared_nodes, ElementsAre(0, 0));
  EXPECT_EQ(result.shared_edges.size(), 0);
  ASSERT_EQ(result.shared_node_weights.size(), 2);
  EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1));
  EXPECT_EQ(result.shared_edge_weights.size(), 0);
}

TEST(LocalGraphExtractionTest, extract_local_edge_from_isolated_edges_graph_1) {
  auto graph = make_isolated_edges_graph(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto result = extract_local_subgraphs(p_graph);

  ASSERT_EQ(result.shared_nodes.size(), 2);
  EXPECT_THAT(result.shared_nodes, ElementsAre(1, 2));
  ASSERT_EQ(result.shared_edges.size(), 2);
  EXPECT_THAT(result.shared_edges, ElementsAre(1, 0));
  ASSERT_EQ(result.shared_node_weights.size(), 2);
  EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1));
  ASSERT_EQ(result.shared_edge_weights.size(), 2);
  EXPECT_THAT(result.shared_edge_weights, ElementsAre(1, 1));
}

TEST(LocalGraphExtractionTest, extract_empty_graph) {
  auto graph = make_empty_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto result = extract_local_subgraphs(p_graph);

  EXPECT_EQ(result.shared_nodes.size(), 0);
  EXPECT_EQ(result.shared_edges.size(), 0);
  EXPECT_EQ(result.shared_node_weights.size(), 0);
  EXPECT_EQ(result.shared_edge_weights.size(), 0);
}

TEST(LocalGraphExtractionTest, extract_circle_graph) {
  auto graph = make_circle_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto result = extract_local_subgraphs(p_graph);

  ASSERT_EQ(result.shared_nodes.size(), 1);
  EXPECT_THAT(result.shared_nodes, ElementsAre(0));
  EXPECT_EQ(result.shared_edges.size(), 0);
  ASSERT_EQ(result.shared_node_weights.size(), 1);
  EXPECT_THAT(result.shared_node_weights, ElementsAre(1));
  EXPECT_EQ(result.shared_edge_weights.size(), 0);
}

TEST(LocalGraphExtractionTest, extract_local_triangles) {
  auto graph = make_circle_clique_graph(3);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto result = extract_local_subgraphs(p_graph);

  ASSERT_EQ(result.shared_nodes.size(), 3);
  EXPECT_THAT(result.shared_nodes, ElementsAre(2, 4, 6));
  ASSERT_EQ(result.shared_edges.size(), 6);
  EXPECT_THAT(result.shared_edges, UnorderedElementsAre(1, 2, 0, 2, 0, 1));
  ASSERT_EQ(result.shared_node_weights.size(), 3);
  EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1, 1));
  ASSERT_EQ(result.shared_edge_weights.size(), 6);
  EXPECT_THAT(result.shared_edge_weights, ElementsAre(1, 1, 1, 1, 1, 1));
}

////////////////////////////////////////////////////////////////////////////////
/// Extract global block induced subgraphs
////////////////////////////////////////////////////////////////////////////////

inline auto extract_global_subgraphs(const DistributedPartitionedGraph &p_graph) {
  return graph::extract_and_scatter_block_induced_subgraphs(p_graph).subgraphs;
}

// One isolated node on each PE, no edges at all
TEST(GlobalGraphExtractionTest, extract_local_node_from_isolated_nodes_graph_1) {
  auto graph = make_isolated_nodes_graph(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should get one block
  ASSERT_EQ(subgraphs.size(), 1);

  // ech block should be a single node without any neighbors
  const auto &subgraph = subgraphs.front();
  EXPECT_EQ(subgraph.n(), 1);
  EXPECT_EQ(subgraph.m(), 0);
  EXPECT_EQ(subgraph.total_node_weight(), 1);
  EXPECT_EQ(subgraph.total_edge_weight(), 0);
}

// Two isolated nodes on each PE, no edges at all
TEST(GlobalGraphExtractionTest, extract_local_nodes_from_isolated_nodes_graph_2) {
  auto graph = make_isolated_nodes_graph(2);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should get one block
  ASSERT_EQ(subgraphs.size(), 1);

  // each block should consist of two isolated nodes
  const auto &subgraph = subgraphs.front();
  EXPECT_EQ(subgraph.n(), 2);
  EXPECT_EQ(subgraph.m(), 0);
  EXPECT_EQ(subgraph.total_node_weight(), 2);
  EXPECT_EQ(subgraph.total_edge_weight(), 0);
}

// Test empty blocks
TEST(GlobalGraphExtractionTest, extract_empty_graphs) {
  auto graph = make_empty_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  // still expect one (empty) block pe PE
  ASSERT_EQ(subgraphs.size(), 1);

  const auto &subgraph = subgraphs.front();
  EXPECT_EQ(subgraph.n(), 0);
  EXPECT_EQ(subgraph.m(), 0);
  EXPECT_EQ(subgraph.total_node_weight(), 0);
  EXPECT_EQ(subgraph.total_edge_weight(), 0);
}

// Test with local egdes
TEST(GlobalGraphExtractionTest, extract_local_edge) {
  auto graph = make_isolated_edges_graph(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should get one block
  ASSERT_EQ(subgraphs.size(), 1);

  // the block should contain a single edge
  const auto &subgraph = subgraphs.front();
  ASSERT_EQ(subgraph.n(), 2);
  EXPECT_EQ(subgraph.degree(0), 1);
  EXPECT_EQ(subgraph.degree(1), 1);
  EXPECT_EQ(subgraph.m(), 2);
}

// Test with 10 local egdes
TEST(GlobalGraphExtractionTest, extract_local_edges) {
  auto graph = make_isolated_edges_graph(10);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should still get one block
  ASSERT_EQ(subgraphs.size(), 1);

  // the block should contain 10 edges
  const auto &subgraph = subgraphs.front();
  ASSERT_EQ(subgraph.n(), 20);
  EXPECT_EQ(subgraph.m(), 20);

  for (const NodeID u : subgraph.nodes()) {
    EXPECT_EQ(subgraph.degree(u), 1);
    const NodeID neighbor = subgraph.edge_target(subgraph.first_edge(u));
    EXPECT_EQ(subgraph.degree(neighbor), 1);
    const NodeID neighbor_neighbor = subgraph.edge_target(subgraph.first_edge(neighbor));
    EXPECT_EQ(neighbor_neighbor, u);
  }
}

// Test with cut edges: ring across PEs, but there should be still no local
// egdes
TEST(GlobalGraphExtractionTest, extract_local_node_from_circle_graph) {
  auto graph = make_circle_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should still get one block
  ASSERT_EQ(subgraphs.size(), 1);

  // each block should contain a single node
  const auto &subgraph = subgraphs.front();
  ASSERT_EQ(subgraph.n(), 1);
  EXPECT_EQ(subgraph.m(), 0);
}

// Test extracting isolated nodes that are spread across PEs
TEST(GlobalGraphExtractionTest, extract_distributed_isolated_nodes) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  // create graph with one local node for each PE
  auto graph = make_isolated_nodes_graph(size);
  std::vector<BlockID> partition(size);
  std::iota(partition.begin(), partition.end(), 0);
  auto p_graph = make_partitioned_graph(graph, static_cast<BlockID>(size), partition);

  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should get one block
  ASSERT_EQ(subgraphs.size(), 1);
  const auto &subgraph = subgraphs.front();

  // ... with size isolated nodes
  ASSERT_EQ(subgraph.n(), size);
  ASSERT_EQ(subgraph.m(), 0);
}

void expect_circle(const shm::Graph &graph) {
  // Catch special case with just 2 nodes: expect a single edge between the two nodes
  if (graph.n() == 2) {
    EXPECT_EQ(graph.degree(0), 1);
    EXPECT_EQ(graph.degree(1), 1);
    EXPECT_EQ(graph.edge_target(graph.first_edge(0)), 1);
    EXPECT_EQ(graph.edge_target(graph.first_edge(1)), 0);
    return;
  }

  NodeID num_nodes_in_circle = 1;
  NodeID start = 0;
  NodeID prev = start;
  NodeID cur = graph.degree(start) > 0 ? graph.edge_target(graph.first_edge(start)) : start;

  while (cur != start) {
    EXPECT_EQ(graph.degree(cur), 2);

    const NodeID neighbor1 = graph.edge_target(graph.first_edge(cur));
    const NodeID neighbor2 = graph.edge_target(graph.first_edge(cur) + 1);
    EXPECT_TRUE(neighbor1 == prev || neighbor2 == prev);

    // move to next node
    prev = cur;
    cur = (neighbor1 == prev) ? neighbor2 : neighbor1;

    ++num_nodes_in_circle;
  }

  EXPECT_EQ(num_nodes_in_circle, graph.n());
}

// Test local clique + global circle extraction, where nodes within a clique
// belong to different blocks
TEST(GlobalGraphExtractionTest, extract_circles_from_clique_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  auto graph = make_circle_clique_graph(size);
  std::vector<BlockID> partition(size);
  std::iota(partition.begin(), partition.end(), 0);
  auto p_graph = make_partitioned_graph(graph, static_cast<BlockID>(size), partition);

  auto subgraphs = extract_global_subgraphs(p_graph);

  // each PE should still get one block
  ASSERT_EQ(subgraphs.size(), 1);
  const auto &subgraph = subgraphs.front();

  // each block should be a circle
  ASSERT_EQ(subgraph.n(), size);

  if (size == 1) {
    EXPECT_EQ(subgraph.m(), 0);
  } else if (size == 2) {
    EXPECT_EQ(subgraph.m(), 2);
  } else {
    EXPECT_EQ(subgraph.m(), 2 * size);
    expect_circle(subgraph);
  }
}

// Test extracting two blocks per PE, each block with a isolated node
TEST(GlobalGraphExtractionTest, extract_two_isolated_node_blocks_per_pe) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  auto graph = make_isolated_nodes_graph(2);
  auto p_graph = make_partitioned_graph(
      graph, 2 * size, {static_cast<BlockID>(2 * rank), static_cast<BlockID>(2 * rank + 1)}
  );
  auto subgraphs = extract_global_subgraphs(p_graph);

  // two blocks per PE
  ASSERT_EQ(subgraphs.size(), 2);

  // each containing a single block
  for (const auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph.n(), 1);
    EXPECT_EQ(subgraph.m(), 0);
  }
}

// Test extracting two blocks, both containing a circle
TEST(GlobalGraphExtractionTest, extract_two_blocks_from_clique_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  auto graph = make_circle_clique_graph(2 * size); // two nodes per PE
  std::vector<BlockID> local_partition(2 * size);
  for (const NodeID u : graph.nodes()) {
    local_partition[u] = u;
  }
  auto p_graph = make_partitioned_graph(graph, 2 * size, local_partition);

  auto subgraphs = extract_global_subgraphs(p_graph);

  // two blocks per PE
  ASSERT_EQ(subgraphs.size(), 2);

  // each containing a circle
  for (const auto &subgraph : subgraphs) {
    EXPECT_EQ(subgraph.n(), size);

    if (size == 1) {
      EXPECT_EQ(subgraph.m(), 0);
    } else if (size == 2) {
      EXPECT_EQ(subgraph.m(), 2); // just two nodes with an edge between them
    } else {
      EXPECT_EQ(subgraph.m(), 2 * size);
      expect_circle(subgraph);
    }
  }
}

// Test node weights
TEST(GlobalGraphExtractionTest, extract_node_weights_in_circle_clique_graph) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  // create clique/circle graph with rank as node weight
  auto graph = make_circle_clique_graph(2 * size);
  std::vector<std::pair<NodeID, NodeWeight>> node_weights;
  std::vector<BlockID> local_partition;
  for (const NodeID u : graph.nodes()) {
    node_weights.emplace_back(u, rank + 1);
    local_partition.push_back(u);
  }
  for (const NodeID u : graph.ghost_nodes()) {
    node_weights.emplace_back(u, graph.ghost_owner(u) + 1);
  }
  graph = change_node_weights(std::move(graph), node_weights);
  auto p_graph = make_partitioned_graph(graph, 2 * size, local_partition);
  auto subgraphs = extract_global_subgraphs(p_graph);

  ASSERT_EQ(subgraphs.size(), 2);

  std::vector<int> weights(size + 1);

  for (const auto &subgraph : subgraphs) {
    for (const NodeID u : subgraph.nodes()) {
      const NodeWeight weight = subgraph.node_weight(u);
      EXPECT_LT(weight, size + 1);
      EXPECT_LE(weights[weight], 2);
      ++weights[weight];
    }
  }

  for (std::size_t i = 1; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], 2);
  }
}

// Test edge weights
TEST(GlobalGraphExtractionTest, extract_local_edge_weights_in_circle_clique_graph) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  // create clique/circle graph with rank as node weight
  auto graph = make_circle_clique_graph(2);

  std::vector<std::tuple<EdgeID, EdgeID, EdgeWeight>> edge_weights;
  edge_weights.emplace_back(0, 1, rank);
  edge_weights.emplace_back(1, 0, rank);

  graph = change_edge_weights_by_endpoints(std::move(graph), edge_weights);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto subgraphs = extract_global_subgraphs(p_graph);

  ASSERT_EQ(subgraphs.size(), 1);
  auto &subgraph = subgraphs.front();

  ASSERT_EQ(subgraph.n(), 2);
  ASSERT_EQ(subgraph.m(), 2);
  EXPECT_EQ(subgraph.edge_weight(0), rank);
  EXPECT_EQ(subgraph.edge_weight(1), rank);
}

// Test copying subgraph partition back to the distributed graph: one isolated
// nodes that is not migrated
TEST(GlobalGraphExtractionTest, project_isolated_nodes_1_partition) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  auto graph = make_isolated_nodes_graph(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);

  auto result = graph::extract_and_scatter_block_induced_subgraphs(p_graph);
  auto &subgraphs = result.subgraphs;

  // one block with one node -> assign to block 0
  auto &subgraph = subgraphs.front();

  std::vector<shm::PartitionedGraph> p_subgraphs;
  StaticArray<BlockID> partition(1);
  partition[0] = 0;
  p_subgraphs.emplace_back(subgraph, 1, std::move(partition));

  // Copy back to p_graph
  p_graph = graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

  EXPECT_EQ(p_graph.k(), size); // k should not have changed
  ASSERT_EQ(p_graph.n(), 1);
  EXPECT_EQ(p_graph.block(0), rank); // partition should not have changed
}

// ... test with two nodes
TEST(GlobalGraphExtractionTest, project_isolated_nodes_2_partition) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  auto graph = make_isolated_nodes_graph(2);
  auto p_graph = make_partitioned_graph_by_rank(graph);

  auto result = graph::extract_and_scatter_block_induced_subgraphs(p_graph);
  auto &subgraphs = result.subgraphs;

  // one block with one node -> assign to block 0
  auto &subgraph = subgraphs.front();

  std::vector<shm::PartitionedGraph> p_subgraphs;
  StaticArray<BlockID> partition(2);
  partition[0] = 0;
  partition[1] = 1;
  p_subgraphs.emplace_back(subgraph, 2, std::move(partition));

  // Copy back to p_graph
  p_graph = graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

  EXPECT_EQ(p_graph.k(), 2 * size); // k should not have doubled
  ASSERT_EQ(p_graph.n(), 2);
  // We cannot tell which node is in which block, only that one should be in
  // block 0 and one in block 1
  EXPECT_NE(p_graph.block(0), p_graph.block(1));
  EXPECT_EQ(p_graph.block(0) + p_graph.block(1), 2 * rank + 2 * rank + 1);
}

// ... test with clique
TEST(GlobalGraphExtractionTest, project_circle_clique_partition) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  auto graph = make_circle_clique_graph(2 * size); // two nodes per PE

  // Always place two nodes in one partition
  std::vector<BlockID> local_partition(2 * size);
  for (const NodeID u : graph.nodes()) {
    local_partition[u] = u / 2;
  }
  auto p_graph = make_partitioned_graph(graph, size, local_partition);

  // Extract blocks
  auto result = graph::extract_and_scatter_block_induced_subgraphs(p_graph);
  auto &subgraphs = result.subgraphs;
  ASSERT_EQ(subgraphs.size(), 1);
  auto &subgraph = subgraphs.front();

  // Should have 2 * size nodes on each PE
  ASSERT_EQ(subgraph.n(), 2 * size);

  // Assign 2 nodes to a new block
  std::vector<shm::PartitionedGraph> p_subgraphs;
  StaticArray<BlockID> partition(2 * size);
  for (const NodeID u : subgraph.nodes()) {
    partition[u] = u / 2;
  }
  p_subgraphs.emplace_back(subgraph, static_cast<BlockID>(size), std::move(partition));

  // Copy back to p_graph
  p_graph = graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

  // Should have size * (size / 2) blocks now
  ASSERT_EQ(p_graph.k(), size * size);

  for (const NodeID u : p_graph.nodes()) {
    EXPECT_EQ(p_graph.block(u), (u / 2) * size + rank);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Extract global block induced subgraphs with fewer blocks than PEs
////////////////////////////////////////////////////////////////////////////////

TEST(GlobalGraphExtractionBlockAssignment, test_first_block_computation_P1_k1) {
  graph::BlockExtractionOffsets offsets(1, 1);
  EXPECT_EQ(offsets.first_block_on_pe(0), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(0), 1);
}

TEST(GlobalGraphExtractionBlockAssignment, test_first_block_computation_P2_k1) {
  graph::BlockExtractionOffsets offsets(2, 1);
  EXPECT_EQ(offsets.first_block_on_pe(0), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(0), 1);
  EXPECT_EQ(offsets.first_block_on_pe(1), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(1), 1);
}

TEST(GlobalGraphExtractionBlockAssignment, test_first_block_computation_P2_k2) {
  graph::BlockExtractionOffsets offsets(2, 2);
  EXPECT_EQ(offsets.first_block_on_pe(0), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(0), 1);
  EXPECT_EQ(offsets.first_block_on_pe(1), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(1), 1);
}

TEST(GlobalGraphExtractionBlockAssignment, test_first_block_computation_P3_k2) {
  graph::BlockExtractionOffsets offsets(3, 2);
  EXPECT_EQ(offsets.first_block_on_pe(0), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(0), 1);
  EXPECT_EQ(offsets.first_block_on_pe(1), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(1), 1);
  EXPECT_EQ(offsets.first_block_on_pe(2), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(2), 1);
}

TEST(GlobalGraphExtractionBlockAssignment, test_first_block_computation_P7_k2) {
  graph::BlockExtractionOffsets offsets(7, 2);
  EXPECT_EQ(offsets.first_block_on_pe(0), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(0), 1);
  EXPECT_EQ(offsets.first_block_on_pe(1), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(1), 1);
  EXPECT_EQ(offsets.first_block_on_pe(2), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(2), 1);
  EXPECT_EQ(offsets.first_block_on_pe(3), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(3), 1);
  EXPECT_EQ(offsets.first_block_on_pe(4), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(4), 1);
  EXPECT_EQ(offsets.first_block_on_pe(5), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(5), 1);
  EXPECT_EQ(offsets.first_block_on_pe(6), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(6), 1);
}

TEST(GlobalGraphExtractionBlockAssignment, test_first_block_computation_P7_k3) {
  graph::BlockExtractionOffsets offsets(7, 3);
  EXPECT_EQ(offsets.first_block_on_pe(0), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(0), 1);
  EXPECT_EQ(offsets.first_block_on_pe(1), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(1), 1);
  EXPECT_EQ(offsets.first_block_on_pe(2), 0);
  EXPECT_EQ(offsets.num_blocks_on_pe(2), 1);
  EXPECT_EQ(offsets.first_block_on_pe(3), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(3), 1);
  EXPECT_EQ(offsets.first_block_on_pe(4), 1);
  EXPECT_EQ(offsets.num_blocks_on_pe(4), 1);
  EXPECT_EQ(offsets.first_block_on_pe(5), 2);
  EXPECT_EQ(offsets.num_blocks_on_pe(5), 1);
  EXPECT_EQ(offsets.first_block_on_pe(6), 2);
  EXPECT_EQ(offsets.num_blocks_on_pe(6), 1);
}

TEST(GlobalGraphExtractionTest, extract_from_circle_clique_graph_fewer_blocks_than_pes) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  auto graph = make_circle_clique_graph(size / 2);

  std::vector<BlockID> local_partition(size / 2);
  std::iota(local_partition.begin(), local_partition.end(), 0);

  // Use global node IDs as node weights
  std::vector<std::pair<NodeID, NodeWeight>> node_weights;
  for (const NodeID u : graph.all_nodes()) {
    node_weights.emplace_back(u, graph.local_to_global_node(u) + 1);
  }
  graph = change_node_weights(std::move(graph), node_weights);

  auto p_graph = make_partitioned_graph(graph, size / 2, local_partition);
  auto subgraphs = extract_global_subgraphs(p_graph);

  if (size == 1) {
    EXPECT_TRUE(subgraphs.empty());
  } else {
    ASSERT_EQ(subgraphs.size(), 1);
    auto &subgraph = subgraphs.front();

    // Check node weights
    graph::BlockExtractionOffsets offsets(size, p_graph.k());
    const BlockID my_block = offsets.first_block_on_pe(rank);
    std::vector<bool> seen_weight(graph.global_n());
    NodeID seen_weights = 0;
    for (const NodeID u : subgraph.nodes()) {
      const NodeWeight weight = subgraph.node_weight(u);
      ASSERT_LT(weight - 1, graph.global_n());
      EXPECT_FALSE(seen_weight[weight - 1]);
      seen_weight[weight - 1] = true;
      ++seen_weights;
    }
    EXPECT_EQ(seen_weights, size);
    for (NodeID u = my_block; u < graph.global_n(); u += size / 2) {
      EXPECT_TRUE(seen_weight[u]) << u;
    }

    // Check topology
    EXPECT_EQ(subgraph.n(), size);
    if (size == 2) {
      EXPECT_EQ(subgraph.m(), 2);
    } else {
      EXPECT_EQ(subgraph.m(), 2 * size);
      expect_circle(subgraph);
    }
  }
}

TEST(GlobalGraphExtractionTest, project_from_circle_clique_graph_less_pes_than_blocks) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  if (size % 2 != 0) {
    return;
  }

  auto graph = make_circle_clique_graph(size / 2);

  // k = size/2
  std::vector<BlockID> local_partition(size / 2);
  std::iota(local_partition.begin(), local_partition.end(), 0);

  auto p_graph = make_partitioned_graph(graph, size / 2, local_partition);
  auto result = graph::extract_and_scatter_block_induced_subgraphs(p_graph);
  auto &subgraphs = result.subgraphs;
  ASSERT_EQ(subgraphs.size(), 1);
  auto &subgraph = subgraphs.front();

  // Create one bad and one good partition
  // If our block is even, the first one is the good one, otherwise the second
  // one
  StaticArray<BlockID> partition(subgraph.n());
  const bool good_partition = rank % 2 == (rank / 2) % 2;
  if (good_partition) { // Good partition: but everything in the same block for
                        // min cut (ignore balance)
    std::fill(partition.begin(), partition.end(), 0);
  } else { // Bad partition: alternate between blocks
    for (const NodeID u : subgraph.nodes()) {
      partition[u] = u % 2;
    }
  }
  shm::PartitionedGraph p_subgraph(subgraph, 2, std::move(partition));
  if (good_partition) {
    EXPECT_EQ(shm::metrics::edge_cut(p_subgraph), 0);
  } else {
    EXPECT_GT(shm::metrics::edge_cut(p_subgraph), 0);
  }
  std::vector<shm::PartitionedGraph> p_subgraphs;
  p_subgraphs.push_back(std::move(p_subgraph));

  // Project back
  p_graph = graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

  EXPECT_EQ(p_graph.k(), size);

  for (const NodeID u : p_graph.nodes()) {
    EXPECT_TRUE(p_graph.block(u) % 2 == 0) << V(u) << V(p_graph.block(u));
  }
}

// Test extracting one block with many PEs = each PE gets a copy of the block
TEST(GlobalGraphExtractionTest, extract_one_block_with_many_pes) {
  const PEID rank = mpi::get_comm_size(MPI_COMM_WORLD);

  auto graph = make_circle_graph();
  auto p_graph = make_partitioned_graph(graph, 1, {0});

  auto result = graph::extract_and_scatter_block_induced_subgraphs(p_graph);
  auto &subgraphs = result.subgraphs;

  ASSERT_EQ(subgraphs.size(), 1);
  expect_circle(subgraphs.front());
}
} // namespace kaminpar::dist
