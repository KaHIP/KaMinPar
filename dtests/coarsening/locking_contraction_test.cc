/*******************************************************************************
* @file:   locking_contraction_test.cc
*
* @author: Daniel Seemaier
* @date:   04.10.21
* @brief:  Unit tests for contracting locked clusters.
******************************************************************************/
#include "dkaminpar/coarsening/locking_clustering_contraction.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dtests/mpi_test.h"

#include <tbb/global_control.h>
#include <unordered_map>

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Eq;
using ::testing::ElementsAre;

namespace dkaminpar::test {
using namespace fixtures3PE;

using Clustering = LockingLpClustering::AtomicClusterArray;

auto contract_clustering(const DistributedGraph &graph, const std::vector<Clustering> &clusterings) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  return ::dkaminpar::graph::contract_locking_clustering(graph, clusterings[rank]);
}

auto contract_clustering(const DistributedGraph &graph, const Clustering &clusterings) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  return ::dkaminpar::graph::contract_locking_clustering(graph, clusterings);
}

TEST_F(DistributedTriangles, TestFullContractionToEachPE) {
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

  for (PEID pe = 0; pe < size; ++pe) {
    const NodeID cluster = pe * size; // 0, 3, 6 -> owned by PE pe
    auto [c_graph, mapping, m_ctx] = contract_clustering(graph, {cluster, cluster, cluster, cluster, cluster, cluster, cluster});

    if (rank == pe) {
      EXPECT_THAT(c_graph.n(), Eq(1));
      EXPECT_THAT(c_graph.node_weights(), ElementsAre(Eq(3)));
    } else {
      EXPECT_THAT(c_graph.n(), Eq(0));
    }
    EXPECT_THAT(c_graph.m(), Eq(0));
    EXPECT_THAT(c_graph.ghost_n(), Eq(0));
    EXPECT_THAT(c_graph.global_n(), Eq(1));
    EXPECT_THAT(c_graph.global_m(), Eq(0));
  }
}

TEST_F(DistributedTriangles, TestHalfContraction) {
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

  Clustering clustering(graph.total_n());
  if (rank == 0) {
    clustering[graph.global_to_local_node(0)] = 0;
    clustering[graph.global_to_local_node(1)] = 0;
    clustering[graph.global_to_local_node(2)] = 0;

    clustering[graph.global_to_local_node(3)] = 0;
    clustering[graph.global_to_local_node(5)] = 0;
    clustering[graph.global_to_local_node(7)] = 6;
    clustering[graph.global_to_local_node(8)] = 6;
  } else if (rank == 1) {
    clustering[graph.global_to_local_node(1)] = 0;
    clustering[graph.global_to_local_node(2)] = 0;

    clustering[graph.global_to_local_node(3)] = 0;
    clustering[graph.global_to_local_node(4)] = 0;
    clustering[graph.global_to_local_node(5)] = 0;

    clustering[graph.global_to_local_node(6)] = 6;
    clustering[graph.global_to_local_node(8)] = 6;
  } else if (rank == 2) {
    clustering[graph.global_to_local_node(0)] = 0;
    clustering[graph.global_to_local_node(2)] = 0;

    clustering[graph.global_to_local_node(4)] = 0;
    clustering[graph.global_to_local_node(5)] = 0;

    clustering[graph.global_to_local_node(6)] = 6;
    clustering[graph.global_to_local_node(7)] = 6;
    clustering[graph.global_to_local_node(8)] = 6;
  }

  auto [c_graph, mapping, m_ctx] = contract_clustering(graph, clustering);

  if (rank == 0) {
    EXPECT_THAT(c_graph.n(), Eq(1));
    EXPECT_THAT(c_graph.m(), Eq(1));
    EXPECT_THAT(c_graph.ghost_n(), Eq(1));
    EXPECT_THAT(c_graph.total_n(), Eq(2));
    EXPECT_THAT(c_graph.node_weights(), ElementsAre(6, 3));
    EXPECT_THAT(c_graph.edge_weights(), ElementsAre(4));
  } else if (rank == 1) {
    EXPECT_THAT(c_graph.n(), Eq(0));
  } else if (rank == 2) {
    EXPECT_THAT(c_graph.n(), Eq(1));
    EXPECT_THAT(c_graph.m(), Eq(1));
    EXPECT_THAT(c_graph.ghost_n(), Eq(1));
    EXPECT_THAT(c_graph.node_weights(), ElementsAre(3, 6));
    EXPECT_THAT(c_graph.edge_weights(), ElementsAre(4));
  }
}
} // namespace dkaminpar::test