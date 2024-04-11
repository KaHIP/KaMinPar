#include <gmock/gmock.h>

#include "tests/dist/distributed_graph_factories.h"
#include "tests/dist/distributed_graph_helpers.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/graphutils/replicator.h"
#include "kaminpar-dist/metrics.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

TEST(GraphReplicationTest, isolated_graph_1) {
  const auto graph = make_isolated_nodes_graph(1);
  const auto rep = replicate_graph(graph, 1);
  ASSERT_TRUE(debug::validate_graph(rep));
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  // Only 1 copy -> graph should stay the same
  EXPECT_EQ(rep.n(), 1);
  EXPECT_EQ(rep.global_n(), size);
  EXPECT_EQ(rep.m(), 0);
}

TEST(GraphReplicationTest, isolated_graph_P) {
  const auto graph = make_isolated_nodes_graph(1);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const auto rep = replicate_graph(graph, size);
  ASSERT_TRUE(debug::validate_graph(rep));

  // size copies -> every PE should own the full graph
  EXPECT_EQ(rep.n(), size);
  EXPECT_EQ(rep.global_n(), size);
  EXPECT_EQ(rep.m(), 0);
}

TEST(GraphReplicationTest, isolated_graph_P_div_2) {
  const auto graph = make_isolated_nodes_graph(1);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  if (size > 2) {
    const auto rep = replicate_graph(graph, size / 2);
    ASSERT_TRUE(debug::validate_graph(rep));

    // EXPECT_EQ(rep.n(), size / 2);
    EXPECT_EQ(rep.global_n(), size);
    EXPECT_EQ(rep.m(), 0);
  }
}

TEST(GraphReplicationTest, triangle_cycle_graph_1) {
  const auto graph = make_circle_clique_graph(3); // triangle on each PE
  const auto rep = replicate_graph(graph, 1);     // replicate 1 == nothing changes
  ASSERT_TRUE(debug::validate_graph(rep));

  EXPECT_EQ(rep.n(), graph.n());
  EXPECT_EQ(rep.global_n(), graph.global_n());
  EXPECT_EQ(rep.m(), graph.m());
  EXPECT_EQ(rep.global_m(), graph.global_m());

  for (const NodeID u : graph.nodes()) {
    EXPECT_EQ(rep.degree(u), graph.degree(u));
  }
}

TEST(GraphReplicationTest, triangle_cycle_graph_P) {
  const auto graph = make_circle_clique_graph(3); // triangle on each PE
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const auto rep = replicate_graph(graph, size); // each PE gets a full copy
  ASSERT_TRUE(debug::validate_graph(rep));

  EXPECT_EQ(rep.n(), rep.global_n());
  EXPECT_EQ(rep.n(), size * 3);
  EXPECT_EQ(rep.m(), rep.global_m());
}

TEST(DistributeBestPartitionTest, triangle_cycle_graph_P) {
  const auto graph = make_circle_clique_graph(3); // triangle on each PE
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const auto rep = replicate_graph(graph, size); // each PE gets a full copy
  ASSERT_TRUE(debug::validate_graph(rep));

  StaticArray<BlockID> partition(rep.n());
  // rank == 0: everything in block 0
  // else: build a bad partition
  if (rank > 0) {
    for (const NodeID u : rep.nodes()) {
      partition[u] = u % 2;
    }
  }

  DistributedPartitionedGraph p_rep(&rep, 2, std::move(partition));

  auto p_graph = distribute_best_partition(graph, std::move(p_rep));

  ASSERT_TRUE(debug::validate_partition(p_graph));
  EXPECT_EQ(metrics::edge_cut(p_graph), 0);
}
} // namespace kaminpar::dist
