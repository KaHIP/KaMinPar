/*******************************************************************************
* @file:   locking_contraction_test.cc
*
* @author: Daniel Seemaier
* @date:   04.10.21
* @brief:  Unit tests for contracting locked clusters.
******************************************************************************/
#include "dkaminpar/algorithm/locking_clustering_contraction.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dtests/mpi_test.h"

#include <tbb/global_control.h>
#include <unordered_map>

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Eq;

namespace dkaminpar::test {
using namespace fixtures3PE;

using Clustering = LockingLpClustering::AtomicClusterArray;

auto contract_clustering(const DistributedGraph &graph, std::vector<Clustering> clusterings) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  return ::dkaminpar::graph::contract_locking_clustering(graph, clusterings[rank]);
}

TEST_F(DistributedTriangles, TestLocalClustering) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  SINGLE_THREADED_TEST;

  //  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  auto [c_graph, mapping,
        m_ctx] = contract_clustering(graph, {{0, 0, 0, 1, 2, 2, 1}, {1, 1, 1, 0, 2, 2, 0}, {2, 2, 2, 0, 1, 1, 0}});

  c_graph.print();

  //  EXPECT_THAT(c_graph.n(), Eq(1));
}
} // namespace dkaminpar::test