/*******************************************************************************
 * @file:   global_contraction_redistribution_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.10.2021
 * @brief:  Unit tests for graph contraction that do not make any assumptions on
 * how the contract graph is distributed across PEs.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_clustering_contraction_redistribution.h"
#include "dkaminpar/coarsening/seq_global_clustering_contraction_redistribution.h"

#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dtests/mpi_test.h"

#include <utility>

using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;

namespace dkaminpar::test {
using namespace fixtures3PE;

using Clustering = coarsening::GlobalClustering;

struct GlobalClusteringContractionRedistribution {
  static auto contract_clustering(const DistributedGraph &graph, const Clustering &clustering) {
    return coarsening::contract_global_clustering_redistribute(graph, clustering);
  }
};

struct SequentialGlobalClusteringContractionRedistribution {
  static auto contract_clustering(const DistributedGraph &graph, const Clustering &clustering) {
    auto [c_graph, c_mapping, m_ctx] =
        coarsening::contract_global_clustering_redistribute_sequential(graph, clustering, {});
    return std::make_pair(std::move(c_graph), std::move(c_mapping));
  }
};

template <typename Contractor> struct Typed { Contractor contractor; };
template <typename Contractor> class TrianglesGraph : public DistributedTriangles, public Typed<Contractor> {};
template <typename Contractor>
class EmptyGraph : public DistributedGraphWith9NodesAnd0Edges, public Typed<Contractor> {};
template <typename Contractor> class NullGraph : public DistributedNullGraph, public Typed<Contractor> {};
template <typename Contractor> class PathGraph : public DistributedPathTwoNodesPerPE, public Typed<Contractor> {};

using Contractors =
    ::testing::Types<GlobalClusteringContractionRedistribution, SequentialGlobalClusteringContractionRedistribution>;

TYPED_TEST_SUITE(TrianglesGraph, Contractors);
TYPED_TEST_SUITE(EmptyGraph, Contractors);
TYPED_TEST_SUITE(NullGraph, Contractors);
TYPED_TEST_SUITE(PathGraph, Contractors);

TYPED_TEST(TrianglesGraph, ContractSingletonClusters) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  const auto clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.n(), this->graph.n()); // since the graph is already evenly distributed
  EXPECT_EQ(c_graph.total_n(), this->graph.total_n());
  EXPECT_EQ(c_graph.global_n(), this->graph.global_n());
  EXPECT_EQ(c_graph.global_m(), this->graph.global_m());
  EXPECT_EQ(c_graph.total_node_weight(), this->graph.total_node_weight());
  graph::expect_isomorphic(c_graph, this->graph);
}

TYPED_TEST(TrianglesGraph, FullContractionToPE0) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  auto [c_graph, mapping] = this->contractor.contract_clustering(this->graph, {0, 0, 0, 0, 0, 0, 0});

  if (this->rank == 0) {
    EXPECT_THAT(c_graph.n(), Eq(1));
  }

  EXPECT_THAT(c_graph.m(), Eq(0));
  EXPECT_THAT(c_graph.global_n(), Eq(1));
  EXPECT_THAT(c_graph.global_m(), Eq(0));

  graph::expect_empty(c_graph);
}

TYPED_TEST(TrianglesGraph, FullContractionToEachPE) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph)); // for isomorphism check

  for (PEID pe = 0; pe < this->size; ++pe) {
    const NodeID cluster = pe * this->size; // 0, 3, 6 -> owned by PE pe
    auto [c_graph, mapping] = this->contractor.contract_clustering(
        this->graph, {cluster, cluster, cluster, cluster, cluster, cluster, cluster});

    if (this->rank == 0) {
      EXPECT_THAT(c_graph.n(), Eq(1));
    }

    EXPECT_THAT(c_graph.m(), Eq(0));
    EXPECT_THAT(c_graph.global_n(), Eq(1));
    EXPECT_THAT(c_graph.global_m(), Eq(0));

    graph::expect_empty(c_graph);
  }
}

TYPED_TEST(TrianglesGraph, ContractLocalTriangles) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  Clustering clustering;
  for (const NodeID u : this->graph.all_nodes()) {
    clustering.push_back(this->graph.find_owner_of_global_node(this->graph.local_to_global_node(u)));
  }

  auto [c_graph, mapping] = this->contractor.contract_clustering(this->graph, clustering);

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

TYPED_TEST(TrianglesGraph, ContractTriangleOnOnePE) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  // contract nodes on PE 0 to one node, keep all other nodes in their cluster
  Clustering clustering;
  for (const NodeID u : this->graph.all_nodes()) {
    const auto u_global = this->graph.local_to_global_node(u);
    const PEID u_pe = this->graph.find_owner_of_global_node(u_global);

    if (u_pe == 0) {
      clustering.push_back(0);
    } else {
      clustering.push_back(u_global);
    }
  }

  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

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

TYPED_TEST(TrianglesGraph, ContractTrianglesOnTwoPEs) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  // contract nodes on PE 0 to one node, keep all other nodes in their cluster
  Clustering clustering;
  for (const NodeID u : this->graph.all_nodes()) {
    const auto u_global = this->graph.local_to_global_node(u);
    const PEID u_pe = this->graph.find_owner_of_global_node(u_global);

    if (u_pe == 0) {
      clustering.push_back(0);
    } else if (u_pe == 1) {
      clustering.push_back(1);
    } else {
      clustering.push_back(u_global);
    }
  }

  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

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

TYPED_TEST(TrianglesGraph, ContractRowWise) {
  //  0---1-#-3---4  -- C0 # C1
  //  |\ /  #  \ /|
  //  | 2---#---5 |  -- C2
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |  -- C3
  //  |    / \    |
  //  +---7---6---+  -- C4
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 0, 2, 1, 1, 2, 4, 4, 3});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

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
// Edge cases: no nodes, no edges
//

TYPED_TEST(NullGraph, ContractNullGraph) {
  Clustering clustering;
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 0);
  EXPECT_EQ(c_graph.global_m(), 0);
  EXPECT_EQ(c_graph.n(), 0);
  EXPECT_EQ(c_graph.m(), 0);
  EXPECT_EQ(c_graph.ghost_n(), 0);
  EXPECT_EQ(c_graph.total_n(), 0);
  EXPECT_EQ(c_graph.total_node_weight(), 0);
  EXPECT_EQ(c_graph.global_total_node_weight(), 0);
}

TYPED_TEST(EmptyGraph, ContractEmtpyGraphToOneNode) {
  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 1);
  EXPECT_EQ(c_graph.global_m(), 0);
  EXPECT_EQ(c_graph.m(), 0);
  EXPECT_EQ(c_graph.global_total_node_weight(), this->graph.global_total_node_weight());
}

TYPED_TEST(EmptyGraph, ContractEmptyGraphToOneNodePerPE) {
  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 0, 0, 1, 1, 1, 2, 2, 2});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 3);
  EXPECT_EQ(c_graph.global_m(), 0);
  EXPECT_EQ(c_graph.global_total_node_weight(), this->graph.global_total_node_weight());
  EXPECT_THAT(c_graph.node_weights(), ElementsAre(Eq(3)));
}

//
// Test on graph path where not all PEs are adjacent
//

TYPED_TEST(PathGraph, ContractPathToCluster0) {
  // 0--1-#-2--3-#-4--5
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 0, 0, 0, 0, 0});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 1);
  EXPECT_EQ(c_graph.global_m(), 0);
  EXPECT_EQ(c_graph.global_total_node_weight(), this->graph.global_total_node_weight());
}

TYPED_TEST(PathGraph, ContractEachHalfToOneNode) {
  // 0--1-#-2--3-#-4--5
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 0, 0, 5, 5, 5});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 2);
  EXPECT_EQ(c_graph.global_m(), 2);
  EXPECT_EQ(c_graph.global_total_node_weight(), this->graph.global_total_node_weight());

  graph::expect_isomorphic(c_graph, {{0b000'111, 1, 0b111'000}});
}

TYPED_TEST(PathGraph, ContractMiddlePart) {
  // 0--1-#-2--3-#-4--5
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 1, 2, 2, 4, 5});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 5);
  EXPECT_EQ(c_graph.global_m(), 8);
  EXPECT_EQ(c_graph.global_total_node_weight(), this->graph.global_total_node_weight());

  graph::expect_isomorphic(c_graph, {
                                        {0b00'00'01, 1, 0b00'00'10},
                                        {0b00'00'10, 1, 0b00'11'00},
                                        {0b00'11'00, 1, 0b01'00'00},
                                        {0b01'00'00, 1, 0b10'00'00},
                                    });
}

TYPED_TEST(PathGraph, ContractMiddleOut) {
  // 0--1-#-2--3-#-4--5
  this->graph = graph::assign_node_weight_identifiers(std::move(this->graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 1, 2, 2, 1, 0});
  const auto [c_graph, c_mapping] = this->contractor.contract_clustering(this->graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 3);
  EXPECT_EQ(c_graph.global_m(), 4);
  EXPECT_EQ(c_graph.global_total_node_weight(), this->graph.global_total_node_weight());

  graph::expect_isomorphic(c_graph, {
                                        {0b10'00'01, 2, 0b01'00'10},
                                        {0b01'00'10, 2, 0b00'11'00},
                                    });
}
} // namespace dkaminpar::test