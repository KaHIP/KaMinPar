/*******************************************************************************
 * Unit tests for implementation details of distributed cluster contraction.
 *
 * @file:   internal_cluster_contraction_test.cc
 * @author: Daniel Seemaier
 * @date:   12.09.2023
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/mpi/utils.h"

// Implementation file to be tested
#include "dkaminpar/coarsening/contraction/cluster_contraction.cc"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;
using namespace ::testing;

namespace {
StaticArray<GlobalNodeID> build_cnode_distribution(const GlobalNodeID n) {
  return mpi::build_distribution_from_local_count<GlobalNodeID, StaticArray>(n, MPI_COMM_WORLD);
}
} // namespace

TEST(ClusterReassignmentTest, perfectly_balanced_case) {
  const auto graph = make_isolated_nodes_graph(4);
  const auto cnode_distribution = build_cnode_distribution(2);
  const auto result = compute_assignment_shifts(graph, cnode_distribution, 1.0);
  EXPECT_THAT(result.overload, Each(Eq(0)));
  EXPECT_THAT(result.underload, Each(Eq(0)));
}

TEST(ClusterReassignmentTest, stair_no_limit) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  const auto graph = make_isolated_nodes_graph(size * size);
  const auto cnode_distribution = build_cnode_distribution(2 * (rank + 1));
  const auto result = compute_assignment_shifts(graph, cnode_distribution, 1.0);

  const GlobalNodeID expected = size + 1;

  for (PEID pe = 0; pe < size / 2; ++pe) {
    EXPECT_EQ(result.overload[pe + 1] - result.overload[pe], 0);
    EXPECT_EQ(result.underload[pe + 1] - result.underload[pe], expected - 2 * (pe + 1));
  }
  for (PEID pe = size / 2; pe < size; ++pe) {
    EXPECT_EQ(result.overload[pe + 1] - result.overload[pe], 2 * (pe + 1) - expected);
    EXPECT_EQ(result.underload[pe + 1] - result.underload[pe], 0);
  }
}
} // namespace kaminpar::dist
