/*******************************************************************************
 * @file:   global_contraction_redistribution_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.10.2021
 * @brief:  Unit tests for global contraction with graph redistribution.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_clustering_contraction_redistribution.h"

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dtests/mpi_test.h"

using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;

namespace dkaminpar::test {
using namespace fixtures3PE;

using Clustering = coarsening::GlobalClustering;

auto contract_clustering(const DistributedGraph &graph, const std::vector<Clustering> &clusterings) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  return coarsening::contract_global_clustering_redistribute(graph, clusterings[rank]);
}

auto contract_clustering(const DistributedGraph &graph, const Clustering &clusterings) {
  return coarsening::contract_global_clustering_redistribute(graph, clusterings);
}

TEST_F(DistributedTriangles, TestFullContractionToPE0) {
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

  auto [c_graph, mapping] = contract_clustering(graph, {0, 0, 0, 0, 0, 0, 0});

  if (rank == 0) {
    EXPECT_THAT(c_graph.n(), Eq(1));
    EXPECT_THAT(c_graph.node_weights(), ElementsAre(Eq(9)));
    EXPECT_THAT(c_graph.total_node_weight(), Eq(9));
  }

  EXPECT_THAT(c_graph.m(), Eq(0));
  EXPECT_THAT(c_graph.global_n(), Eq(1));
  EXPECT_THAT(c_graph.global_m(), Eq(0));
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
    auto [c_graph, mapping] =
        contract_clustering(graph, {cluster, cluster, cluster, cluster, cluster, cluster, cluster});

    if (rank == 0) {
      EXPECT_THAT(c_graph.n(), Eq(1));
      EXPECT_THAT(c_graph.node_weights(), ElementsAre(Eq(9)));
      EXPECT_THAT(c_graph.total_node_weight(), Eq(9));
    }

    EXPECT_THAT(c_graph.m(), Eq(0));
    EXPECT_THAT(c_graph.global_n(), Eq(1));
    EXPECT_THAT(c_graph.global_m(), Eq(0));
  }
}

TEST_F(DistributedTriangles, ContractLocalTriangles) {
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

  Clustering clustering;
  for (const NodeID u : graph.all_nodes()) {
    clustering.push_back(graph.find_owner_of_global_node(graph.local_to_global_node(u)));
  }

  const auto [c_graph, mapping] = contract_clustering(graph, clustering);

  EXPECT_THAT(c_graph.n(), Eq(1));
  EXPECT_THAT(c_graph.total_n(), Eq(3));
  EXPECT_THAT(c_graph.ghost_n(), Eq(2));
  EXPECT_THAT(c_graph.node_weights(), ElementsAre(Eq(3), Eq(3), Eq(3)));
  EXPECT_THAT(c_graph.m(), Eq(2));
  EXPECT_THAT(c_graph.edge_weights(), ElementsAre(Eq(2), Eq(2)));
  EXPECT_THAT(c_graph.global_m(), Eq(6));
}

TEST_F(DistributedTriangles, TestTriangleContractionOnOnePE) {
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

  // contract nodes on PE 0 to one node, keep all other nodes in their cluster
  Clustering clustering;
  for (const NodeID u : graph.all_nodes()) {
    const auto u_global = graph.local_to_global_node(u);
    const PEID u_pe = graph.find_owner_of_global_node(u_global);

    if (u_pe == 0) {
      clustering.push_back(0);
    } else {
      clustering.push_back(u_global);
    }
  }

  const auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

  EXPECT_THAT(c_graph.global_n(), Eq(7));
  EXPECT_THAT(c_graph.global_m(), Eq(24));
  EXPECT_THAT(c_graph.node_weights(), Contains(Eq(3)));
  EXPECT_THAT(c_graph.node_weights(), Each(AnyOf(Eq(1), Eq(3))));
  EXPECT_THAT(c_graph.edge_weights(), Each(Eq(1)));
}

TEST_F(DistributedTriangles, TestTriangleContractionOnTwoPEs) {
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

  // contract nodes on PE 0 to one node, keep all other nodes in their cluster
  Clustering clustering;
  for (const NodeID u : graph.all_nodes()) {
    const auto u_global = graph.local_to_global_node(u);
    const PEID u_pe = graph.find_owner_of_global_node(u_global);

    if (u_pe == 0) {
      clustering.push_back(0);
    } else if (u_pe == 1) {
      clustering.push_back(1);
    } else {
      clustering.push_back(u_global);
    }
  }

  const auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

  EXPECT_THAT(c_graph.global_n(), Eq(5));
  EXPECT_THAT(c_graph.global_m(), Eq(16));
  EXPECT_THAT(c_graph.node_weights(), Contains(Eq(3)));
  EXPECT_THAT(c_graph.node_weights(), Each(AnyOf(Eq(1), Eq(3))));
}
} // namespace dkaminpar::test