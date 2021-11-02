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

TEST_F(DistributedTriangles, FullContractionToPE0) {
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
  graph = graph::assign_node_weight_identifiers(std::move(graph));

  auto [c_graph, mapping] = contract_clustering(graph, {0, 0, 0, 0, 0, 0, 0});

  if (rank == 0) {
    EXPECT_THAT(c_graph.n(), Eq(1));
  }

  EXPECT_THAT(c_graph.m(), Eq(0));
  EXPECT_THAT(c_graph.global_n(), Eq(1));
  EXPECT_THAT(c_graph.global_m(), Eq(0));

  graph::expect_isomorphic(c_graph, {});
}

TEST_F(DistributedTriangles, FullContractionToEachPE) {
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
  graph = graph::assign_node_weight_identifiers(std::move(graph)); // for isomorphism check

  for (PEID pe = 0; pe < size; ++pe) {
    const NodeID cluster = pe * size; // 0, 3, 6 -> owned by PE pe
    auto [c_graph, mapping] =
        contract_clustering(graph, {cluster, cluster, cluster, cluster, cluster, cluster, cluster});

    if (rank == 0) {
      EXPECT_THAT(c_graph.n(), Eq(1));
    }

    EXPECT_THAT(c_graph.m(), Eq(0));
    EXPECT_THAT(c_graph.global_n(), Eq(1));
    EXPECT_THAT(c_graph.global_m(), Eq(0));

    graph::expect_isomorphic(c_graph, {});
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
  graph = graph::assign_node_weight_identifiers(std::move(graph));

  Clustering clustering;
  for (const NodeID u : graph.all_nodes()) {
    clustering.push_back(graph.find_owner_of_global_node(graph.local_to_global_node(u)));
  }

  auto [c_graph, mapping] = contract_clustering(graph, clustering);

  EXPECT_THAT(c_graph.n(), Eq(1));
  EXPECT_THAT(c_graph.total_n(), Eq(3));
  EXPECT_THAT(c_graph.ghost_n(), Eq(2));
  EXPECT_THAT(c_graph.m(), Eq(2));
  EXPECT_THAT(c_graph.edge_weights(), ElementsAre(Eq(2), Eq(2)));
  EXPECT_THAT(c_graph.global_m(), Eq(6));

  graph::expect_isomorphic(c_graph, {
                                        {0b000'000'111, 2, 0b000'111'000},
                                        {0b000'000'111, 2, 0b111'000'000},
                                        {0b000'111'000, 2, 0b111'000'000},
                                    });
}

TEST_F(DistributedTriangles, ContractTriangleOnOnePE) {
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
  graph = graph::assign_node_weight_identifiers(std::move(graph));

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
  EXPECT_THAT(c_graph.edge_weights(), Each(Eq(1)));

  graph::expect_isomorphic(c_graph, {
                                        {0b000'000'111, 1, 0b000'001'000},
                                        {0b000'000'111, 1, 0b000'100'000},
                                        {0b000'000'111, 1, 0b100'000'000},
                                        {0b000'000'111, 1, 0b010'000'000},
                                        {0b100'000'000, 1, 0b010'000'000},
                                        {0b100'000'000, 1, 0b001'000'000},
                                        {0b010'000'000, 1, 0b001'000'000},
                                        {0b000'001'000, 1, 0b000'010'000},
                                        {0b000'001'000, 1, 0b000'100'000},
                                        {0b000'100'000, 1, 0b000'010'000},
                                        {0b000'100'000, 1, 0b100'000'000},
                                        {0b000'010'000, 1, 0b001'000'000},
                                    });
}

TEST_F(DistributedTriangles, ContractTrianglesOnTwoPEs) {
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
  graph = graph::assign_node_weight_identifiers(std::move(graph));

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

  graph::expect_isomorphic(c_graph, {
                                        {0b000'000'111, 2, 0b000'111'000}, // [012] -- [345]
                                        {0b100'000'000, 1, 0b010'000'000}, // [8] -- [7]
                                        {0b100'000'000, 1, 0b001'000'000}, // [8] -- [6]
                                        {0b010'000'000, 1, 0b001'000'000}, // [7] -- [6]
                                        {0b100'000'000, 1, 0b000'000'111}, // [8] -- [012]
                                        {0b100'000'000, 1, 0b000'111'000}, // [8] -- [345]
                                        {0b010'000'000, 1, 0b000'000'111}, // [7] -- [012]
                                        {0b001'000'000, 1, 0b000'111'000}, // [6] -- [345]
                                    });
}

TEST_F(DistributedTriangles, ContractRowWise) {
  //  0---1-#-3---4  -- C0 # C1
  //  |\ /  #  \ /|
  //  | 2---#---5 |  -- C2
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |  -- C3
  //  |    / \    |
  //  +---7---6---+  -- C4
  SINGLE_THREADED_TEST;
  graph = graph::assign_node_weight_identifiers(std::move(graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(graph, {0, 0, 2, 1, 1, 2, 4, 4, 3});
  const auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

  EXPECT_THAT(c_graph.global_n(), Eq(5));
  EXPECT_THAT(c_graph.global_m(), Eq(14));

  graph::expect_isomorphic(c_graph, {
                                        {0b000'000'011, 1, 0b000'011'000}, // [01] -- [34]
                                        {0b000'000'011, 2, 0b000'100'100}, // [01] -- [25]
                                        {0b000'011'000, 2, 0b000'100'100}, // [34] -- [25]
                                        {0b000'000'011, 1, 0b011'000'000}, // [01] -- [67]
                                        {0b000'011'000, 1, 0b011'000'000}, // [34] -- [67]
                                        {0b000'100'100, 2, 0b100'000'000}, // [25] -- [8]
                                        {0b011'000'000, 2, 0b100'000'000}, // [67] -- [8]
                                    });
}

//
// Edge case: empty graph
//

TEST_F(DistributedEmptyGraph, ContractEmptyGraph) {
  Clustering clustering;
  const auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 0);
  EXPECT_EQ(c_graph.global_m(), 0);
}

} // namespace dkaminpar::test