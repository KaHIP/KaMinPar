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

namespace dkaminpar::test {
//  0-1 # 2-3
// ###########
//     4-5
class DistributedEdgesFixture : public MpiTestFixture {
protected:
  void SetUp() override {
    MpiTestFixture::SetUp();

    std::tie(size, rank) = mpi::get_comm_info(MPI_COMM_WORLD);
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    scalable_vector<GlobalNodeID> node_distribution{0, 2, 4, 6};
    const GlobalNodeID global_n = 6;
    const GlobalEdgeID global_m = 6;

    n0 = 2 * rank;
    graph = graph::Builder()
                .initialize(global_n, global_m, rank, std::move(node_distribution))
                .create_node(1)
                .create_edge(1, n0 + 1)
                .create_node(1)
                .create_edge(1, n0)
                .finalize();
  }

  DistributedGraph graph;
  GlobalNodeID n0;
};

//  0---1-#-3---4
//  |\ /  #  \ /|
//  | 2---#---5 |
//  |  \  #  /  |
// ###############
//  |    \ /    |
//  |     8     |
//  |    / \    |
//  +---7---6---+
class DistributedTrianglesFixture : public MpiTestFixture {
protected:
  void SetUp() override {
    MpiTestFixture::SetUp();

    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    scalable_vector<GlobalNodeID> node_distribution{0, 3, 6};
    const GlobalNodeID global_n = 9;
    const GlobalEdgeID global_m = 30;

    n0 = 3 * rank;
    graph = graph::Builder{}
                .initialize(global_n, global_m, rank, std::move(node_distribution))
                .create_node(1)
                .create_edge(1, n0 + 1)
                .create_edge(1, n0 + 2)
                .create_edge(1, prev(n0))
                .create_node(1)
                .create_edge(1, n0)
                .create_edge(1, n0 + 2)
                .create_edge(1, next(n0))
                .create_node(1)
                .create_edge(1, n0)
                .create_edge(1, n0 + 1)
                .create_edge(1, next(n0 + 2, 3))
                .create_edge(1, prev(n0 + 2, 3))
                .finalize();
  }

  DistributedGraph graph;
  GlobalNodeID n0;

private:
  GlobalNodeID next(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u + step) % n;
  }

  GlobalNodeID prev(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u < step) ? n + u - step : u - step;
  }
};

TEST_F(DistributedEdgesFixture, ContractingEdgesSimultaneouslyWorks) {
  EXPECT_EQ(graph.n(), 2);
  EXPECT_EQ(graph.m(), 2);
  EXPECT_EQ(graph.global_n(), 6);
  EXPECT_EQ(graph.global_m(), 6);

  // contract each edge
  auto [c_graph, mapping, m_ctx] = graph::contract_local_clustering(graph, {0, 0});

  EXPECT_EQ(c_graph.n(), 1);
  EXPECT_EQ(c_graph.m(), 0);
  EXPECT_EQ(c_graph.global_n(), 3);
  EXPECT_EQ(c_graph.global_m(), 0);
}
} // namespace dkaminpar::test