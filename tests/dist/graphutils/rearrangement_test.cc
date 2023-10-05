/*******************************************************************************
 * @file:   graph_rearrangement_test.cc
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 * @brief:  Unit tests for rearranging distributed graphs by node degree.
 ******************************************************************************/
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tests/dist/distributed_graph_factories.h"

#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/graphutils/rearrangement.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

TEST(GraphRearrangementTest, sort_path_by_degree_buckets) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  auto graph = make_path(2); // two nodes per PE
  auto sorted_graph = graph::rearrange_by_degree_buckets(std::move(graph));

  // Check weights
  EXPECT_EQ(sorted_graph.total_node_weight(), 2);
  for (const NodeID u : sorted_graph.nodes()) {
    EXPECT_EQ(sorted_graph.node_weight(u), 1);
  }

  if (size == 1) {
    for (const NodeID u : sorted_graph.nodes()) {
      EXPECT_EQ(sorted_graph.degree(u), 1);
    }
  } else if (rank == 0 || rank + 1 == size) {
    EXPECT_EQ(sorted_graph.degree(0), 1);
    EXPECT_EQ(sorted_graph.degree(1), 2);
  } else {
    for (const NodeID u : sorted_graph.nodes()) {
      EXPECT_EQ(sorted_graph.degree(u), 2);
    }
  }
}
} // namespace kaminpar::dist
