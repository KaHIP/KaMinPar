/*******************************************************************************
 * Unit tests for the BFS batch-extraction algorithm.
 *
 * @file:   bfs_extractor_test.cc
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dist/distributed_graph_factories.h"
#include "tests/dist/distributed_graph_helpers.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/graphutils/bfs_extractor.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::dist::graph {
using namespace kaminpar::dist::testing;

std::pair<std::unique_ptr<shm::Graph>, std::unique_ptr<shm::PartitionedGraph>> extract_bfs_subgraph(
    DistributedPartitionedGraph &p_graph, const PEID hops, const std::vector<NodeID> &seed_nodes
) {
  BfsExtractor extractor(p_graph.graph());
  extractor.initialize(p_graph);
  extractor.set_max_hops(hops);
  auto result = extractor.extract(seed_nodes);
  return {std::move(result.graph), std::move(result.p_graph)};
}

TEST(BfsExtractor, empty_seeds_in_empty_graph) {
  auto graph = make_empty_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 2, {});
  EXPECT_EQ(bfs_graph->n(), p_graph.k());
  EXPECT_EQ(bfs_graph->m(), 0);
}

TEST(BfsExtractor, empty_seeds_in_nonempty_graph) {
  auto graph = make_path(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 2, {});
  EXPECT_EQ(bfs_graph->n(), p_graph.k());
  EXPECT_EQ(bfs_graph->m(), 0);
}

TEST(BfsExtractor, zero_hops_zero_seeds_in_empty_graph) {
  auto graph = make_empty_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 0, {});
  EXPECT_EQ(bfs_graph->n(), p_graph.k());
  EXPECT_EQ(bfs_graph->m(), 0);
}

TEST(BfsExtractor, zero_hops_zero_seeds_in_nonempty_graph) {
  auto graph = make_path(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 0, {});
  EXPECT_EQ(bfs_graph->n(), p_graph.k());
  EXPECT_EQ(bfs_graph->m(), 0);
}

TEST(BfsExtractor, zero_hops_in_path_1_graph) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  auto graph = make_path(1);
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 0, {0});
  EXPECT_EQ(bfs_graph->n(), 1 + p_graph.k());
  EXPECT_EQ(p_bfs_graph->block(0), rank);

  if (size == 1) {
    EXPECT_EQ(bfs_graph->m(), 0);
  } else if (size == 2) {
    EXPECT_EQ(bfs_graph->m(), 1); // edge to block pseudo-node
  } else if (size > 2) {
    if (rank == 0 || rank + 1 == size) {
      EXPECT_EQ(bfs_graph->m(), 1); // edge to one block pseudo-nodes
    } else {
      EXPECT_EQ(bfs_graph->m(), 2); // edges to two block pseudo-nodes
    }
  }
}

TEST(BfsExtractor, zero_hops_in_circle_graph) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  auto graph = make_circle_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 0, {0});
  EXPECT_EQ(bfs_graph->n(), 1 + p_graph.k());
  EXPECT_EQ(p_bfs_graph->block(0), rank);

  if (size == 1) {
    EXPECT_EQ(bfs_graph->m(), 0);
  } else if (size == 2) {
    EXPECT_EQ(bfs_graph->m(), 1); // edge to block pseudo-node
  } else if (size > 2) {
    EXPECT_EQ(bfs_graph->m(), 2); // edges to two block pseudo-nodes
  }
}

TEST(BfsExtractor, one_hop_in_circle_graph) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  auto graph = make_circle_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);
  auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, 1, {0});

  if (size == 1) {
    // Graph is just a single node without any edges
    // Thus, expect a single node without any edges
    ASSERT_EQ(bfs_graph->n(), 1 + p_graph.k());
    EXPECT_EQ(bfs_graph->m(), 0);
    EXPECT_EQ(p_bfs_graph->block(0), rank); // == 0
  } else if (size == 2) {
    // Graph consists of two nodes on two PEs with an edge between them
    // Thus, expect the full graph on each PE
    ASSERT_EQ(bfs_graph->n(), 2 + p_graph.k());
    ASSERT_EQ(bfs_graph->m(), 2);
    EXPECT_THAT(p_bfs_graph->block(0), ::testing::AnyOf(0, 1));
    EXPECT_THAT(p_bfs_graph->block(1), ::testing::AnyOf(0, 1));
    EXPECT_NE(p_bfs_graph->block(0), p_bfs_graph->block(1));
  } else if (size == 3) {
    // Graph is a triangle on three PEs: BFS graph should also be a triangle
    ASSERT_EQ(bfs_graph->n(), 3 + p_graph.k());
    ASSERT_EQ(bfs_graph->m(), 6);
    EXPECT_THAT(p_bfs_graph->block(0), ::testing::AnyOf(0, 1, 2));
    EXPECT_THAT(p_bfs_graph->block(1), ::testing::AnyOf(0, 1, 2));
    EXPECT_THAT(p_bfs_graph->block(2), ::testing::AnyOf(0, 1, 2));
    EXPECT_NE(p_bfs_graph->block(0), p_bfs_graph->block(1));
    EXPECT_NE(p_bfs_graph->block(0), p_bfs_graph->block(2));
    EXPECT_NE(p_bfs_graph->block(1), p_bfs_graph->block(2));
    ASSERT_EQ(bfs_graph->degree(0), 2);
    ASSERT_EQ(bfs_graph->degree(1), 2);
    ASSERT_EQ(bfs_graph->degree(2), 2);
    EXPECT_THAT(local_neighbors(*bfs_graph, 0), ::testing::UnorderedElementsAre(1, 2));
    EXPECT_THAT(local_neighbors(*bfs_graph, 1), ::testing::UnorderedElementsAre(0, 2));
    EXPECT_THAT(local_neighbors(*bfs_graph, 2), ::testing::UnorderedElementsAre(0, 1));
  } else if (size > 3) {
    const BlockID prev = static_cast<BlockID>(rank > 0 ? rank - 1 : size - 1);
    const BlockID next = static_cast<BlockID>((rank + 1) % size);

    // Graph is a circle with diameter > 3
    // Thus, expect a path of length 3 + edges to pseudo-block nodes
    ASSERT_EQ(bfs_graph->n(), 3 + p_graph.k());
    EXPECT_THAT(p_bfs_graph->block(0), ::testing::AnyOf(prev, rank, next));
    EXPECT_THAT(p_bfs_graph->block(1), ::testing::AnyOf(prev, rank, next));
    EXPECT_THAT(p_bfs_graph->block(2), ::testing::AnyOf(prev, rank, next));
    EXPECT_NE(p_bfs_graph->block(0), p_bfs_graph->block(1));
    EXPECT_NE(p_bfs_graph->block(0), p_bfs_graph->block(2));
    EXPECT_NE(p_bfs_graph->block(1), p_bfs_graph->block(2));
    EXPECT_EQ(bfs_graph->m(), 6);
  }
}
} // namespace kaminpar::dist::graph
