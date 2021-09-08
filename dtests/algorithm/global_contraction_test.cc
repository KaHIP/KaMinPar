/*******************************************************************************
* This file is part of KaMinPar.
*
* Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
*
* KaMinPar is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* KaMinPar is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
*
******************************************************************************/
#include "dkaminpar/algorithm/distributed_graph_contraction.h"
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"
#include "mpi_test.h"

#include <gmock/gmock.h>

using ::testing::Each;
using ::testing::Eq;
using ::testing::UnorderedElementsAre;

namespace dkaminpar::test {
//  0---#---1
//   \  #  /
// ###########
//     \ /
//      2
class DistributedSingleTriangleFixture : public DistributedGraphFixture {
protected:
  void SetUp() override {
    DistributedGraphFixture::SetUp();
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    n0 = rank;
    graph = graph::Builder{}
                .initialize(3, 6, rank, {0, 1, 2, 3})
                .create_node(1)
                .create_edge(1, prev(n0, 1, 3))
                .create_edge(1, next(n0, 1, 3))
                .finalize();
  }

  GlobalNodeID n0;
  DistributedGraph graph;
};

TEST_F(DistributedSingleTriangleFixture, DistributedSingleTriangleIsAsExpected) {
  EXPECT_EQ(graph.n(), 1);
  EXPECT_EQ(graph.m(), 2);
  EXPECT_EQ(graph.global_n(), 3);
  EXPECT_EQ(graph.global_m(), 6);
  EXPECT_EQ(graph.ghost_n(), 2);
  EXPECT_THAT(graph.node_weights(), Each(Eq(1)));
  EXPECT_THAT(graph.edge_weights(), Each(Eq(1)));
}

TEST_F(DistributedSingleTriangleFixture, ContractingEdgeAcrossPEsWorks) {
  // contract node of PE 1 with node of PE 0, new node owned by PE 0
  // PE 2 is unchanged
  scalable_vector<NodeID> clustering;
  clustering.push_back((rank == 1) ? graph.global_to_local_node(0) : 0);

  auto [c_graph, mapping, m_ctx] = graph::contract_global_clustering(graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 2);
  EXPECT_EQ(c_graph.global_m(), 2);

  if (rank == 0) {
    EXPECT_EQ(c_graph.n(), 1);
    EXPECT_EQ(c_graph.m(), 1);
    EXPECT_THAT(c_graph.edge_weights(), UnorderedElementsAre(2));
    EXPECT_THAT(c_graph.node_weights(), UnorderedElementsAre(2, 1));
    EXPECT_EQ(c_graph.ghost_n(), 1);
    EXPECT_EQ(c_graph.ghost_owner(1), 2); // only neighbor should be on PE 2
  } else if (rank == 1) {
    EXPECT_EQ(c_graph.n(), 0);
    EXPECT_EQ(c_graph.m(), 0);
  } else { // rank == 2
    EXPECT_EQ(c_graph.n(), 1);
    EXPECT_EQ(c_graph.m(), 1);
    EXPECT_THAT(c_graph.edge_weights(), UnorderedElementsAre(2));
    EXPECT_THAT(c_graph.node_weights(), UnorderedElementsAre(1, 2));
    EXPECT_EQ(c_graph.ghost_n(), 1);
    EXPECT_EQ(c_graph.ghost_owner(1), 0); // only neighbor should be on PE 0
  }
}

TEST_F(DistributedSingleTriangleFixture, ContractingTriangleAcrossPEsWorks) {
  scalable_vector<NodeID> clustering;
  clustering.push_back((rank == 0) ? 0 : graph.global_to_local_node(0));

  auto [c_graph, mapping, m_ctx] = graph::contract_global_clustering(graph, clustering);

  EXPECT_EQ(c_graph.global_n(), 1);
  EXPECT_EQ(c_graph.global_m(), 0);
  EXPECT_EQ(c_graph.m(), 0);

  if (rank == 0) {
    EXPECT_EQ(c_graph.n(), 1);
  } else { // rank == 1 || rank == 2
    EXPECT_EQ(c_graph.n(), 0);
  }
}

//  0---1-#-3---4
//  |\ /  #  \ /|
//  | 2---#---5 |
//  |  \  #  /  |
// ###############
//  |    \ /    |
//  |     8     |
//  |    / \    |
//  +---7---6---+
class DistributedTrianglesFixture : public DistributedGraphFixture {
protected:
  void SetUp() override {
    DistributedGraphFixture::SetUp();
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    n0 = 3 * rank;
    graph = graph::Builder{}
                .initialize(9, 30, rank, {0, 3, 6, 9})
                .create_node(1)
                .create_edge(1, n0 + 1)
                .create_edge(1, n0 + 2)
                .create_edge(1, prev(n0, 2, 9))
                .create_node(1)
                .create_edge(1, n0)
                .create_edge(1, n0 + 2)
                .create_edge(1, next(n0 + 1, 2, 9))
                .create_node(1)
                .create_edge(1, n0)
                .create_edge(1, n0 + 1)
                .create_edge(1, next(n0 + 2, 3, 9))
                .create_edge(1, prev(n0 + 2, 3, 9))
                .finalize();
  }

  DistributedGraph graph;
  GlobalNodeID n0;
};

TEST_F(DistributedTrianglesFixture, DistributedTrianglesAreAsExpected) {
  mpi::barrier(MPI_COMM_WORLD);

  EXPECT_EQ(graph.n(), 3);
  EXPECT_EQ(graph.m(), 10); // 2x3 internal edges, 4 edges to ghost nodes
  EXPECT_EQ(graph.ghost_n(), 4);
  EXPECT_EQ(graph.global_n(), 9);
  EXPECT_EQ(graph.global_m(), 30);
  EXPECT_EQ(graph.total_node_weight(), 3);
}
} // namespace dkaminpar::test