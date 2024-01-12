/*******************************************************************************
 * Unit tests for implementation details of distributed cluster contraction.
 *
 * @file:   internal_cluster_contraction_test.cc
 * @author: Daniel Seemaier
 * @date:   12.09.2023
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dist/distributed_graph_factories.h"
#include "tests/dist/distributed_graph_helpers.h"

#include "kaminpar-mpi/utils.h"

// Implementation file to be tested
#include "kaminpar-dist/coarsening/contraction/cluster_contraction.cc"

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
  const auto result = compute_assignment_shifts(graph.node_distribution(), cnode_distribution, 1.0);
  EXPECT_THAT(result.overload, Each(Eq(0)));
  EXPECT_THAT(result.underload, Each(Eq(0)));
}

TEST(ClusterReassignmentTest, stair_no_limit) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  const auto graph = make_isolated_nodes_graph(size * size);
  const auto cnode_distribution = build_cnode_distribution(2 * (rank + 1));
  const auto result = compute_assignment_shifts(graph.node_distribution(), cnode_distribution, 1.0);

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

//
// Regression tests
//

TEST(ClusterReassignmentTest, twitter_2010_64pe_2copy_regression) {
  const auto node_distribution = static_array::create_from<GlobalNodeID>({
      0,     1096,  2130,  3166,  4269,  5493,  6717,  7941,  9019,  10093, 11187,
      12411, 13635, 14763, 15987, 17041, 17957, 19181, 20395, 21619, 22843, 24067,
      25291, 26368, 27462, 28378, 29294, 30209, 31433, 32348, 33482, 34706, 35621,
  });
  const auto cnode_distribution = static_array::create_from<GlobalNodeID>({
      0,     1094,  2128,  3160,  4260,  5484,  6705,  7924,  8986,  10051, 11141,
      12363, 13587, 14663, 15819, 16873, 17774, 18973, 20163, 21370, 22589, 23806,
      25008, 26070, 27127, 28041, 28951, 29860, 31072, 31961, 33095, 34310, 35224,
  });
  const double max_cnode_imbalance = 1.1;

  const auto shifts = compute_assignment_shifts(node_distribution, cnode_distribution, 1.1);

  for (PEID pe = 0; pe < 32; ++pe) {
    const GlobalNodeID my_underload = shifts.underload[pe + 1] - shifts.underload[pe];
    const GlobalNodeID my_size_before = node_distribution[pe + 1] - node_distribution[pe];
    const GlobalNodeID my_size_after = cnode_distribution[pe + 1] - cnode_distribution[pe];
    EXPECT_LE(my_size_after + my_underload, my_size_before) << V(pe);
  }
}

TEST(ClusterReassignmentTest, rgg2d_N7_M11_2pe_regression) {
  const auto node_distribution = static_array::create_from<GlobalNodeID>({0, 57, 128});
  const auto cnode_distribution = static_array::create_from<GlobalNodeID>({0, 57, 128});
  const double max_cnode_imbalance = 1.1;

  const auto shifts = compute_assignment_shifts(node_distribution, cnode_distribution, 1.1);

  for (PEID pe = 0; pe < 2; ++pe) {
    const GlobalNodeID my_underload = shifts.underload[pe + 1] - shifts.underload[pe];
    EXPECT_EQ(my_underload, 0);
  }
}

//[Warning] pe_underload=0, 0, 2, 4, 8, 8, 8, 9, 9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
//14, 14, 14, 18, 22, 23, 23, 23, 23, 23, 23, 23 pe_overload=0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 6,
//6, 6, 6, 6, 7, 9, 9, 9, 12, 14, 16, 16, 16, 16, 16, 18, 21, 21, 21, 24 total_overload=24

TEST(ClusterReassignmentTest, twitter_2010_128pe_4copies_regression) {
  const auto node_distribution = static_array::create_from<GlobalNodeID>(
      {0,     1321,  2622,  3927,  5257,  6702,  8048,  9379,  10725, 12046, 13503,
       14960, 16312, 17651, 19075, 20435, 21792, 23249, 24706, 26163, 27620, 29077,
       30534, 31991, 33134, 34158, 35098, 36038, 37495, 38952, 39892, 40950, 42407}
  );
  const auto cnode_distribution = static_array::create_from<GlobalNodeID>(
      {0,     1321,  2620,  3923,  5249,  6667,  8012,  9341,  10687, 12003, 13460,
       14917, 16267, 17605, 19028, 20388, 21745, 23200, 24656, 26104, 27558, 29015,
       30471, 31927, 33066, 34086, 35025, 35965, 37421, 38878, 39818, 40876, 42333}
  );
  const double max_cnode_imbalance = 1.1;

  const auto shifts = compute_assignment_shifts(node_distribution, cnode_distribution, 1.1);

  for (PEID pe = 0; pe < 32; ++pe) {
    const GlobalNodeID my_underload = shifts.underload[pe + 1] - shifts.underload[pe];
    const GlobalNodeID my_size_before = node_distribution[pe + 1] - node_distribution[pe];
    const GlobalNodeID my_size_after = cnode_distribution[pe + 1] - cnode_distribution[pe];
    EXPECT_LE(my_size_after + my_underload, my_size_before) << V(pe);
  }
}
} // namespace kaminpar::dist
