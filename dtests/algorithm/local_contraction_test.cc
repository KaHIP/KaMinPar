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
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"

#include <gmock/gmock.h>

namespace dkaminpar::test {
//  0-1 # 2-3
// ###########
//     4-5
class DistributedEdgesFixture : public ::testing::Test {
protected:
  void SetUp() override {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    scalable_vector<GlobalNodeID> node_distribution{0, 2, 4};
    const GlobalNodeID global_n = 9;
    const GlobalEdgeID global_m = 10;
    const GlobalNodeID n0 = 2 * rank;

    graph = graph::Builder()
                .initialize(global_n, global_m, rank, std::move(node_distribution))
                .create_node(1)
                .create_edge(1, n0 + 1)
                .create_node(1)
                .create_edge(1, n0)
                .finalize();
  }

  DistributedGraph graph;
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
class DistributedTrianglesFixture : public ::testing::Test {
protected:
  void SetUp() override {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    scalable_vector<GlobalNodeID> node_distribution{0, 3, 6};
    const GlobalNodeID global_n = 9;
    const GlobalEdgeID global_m = 30;
    const GlobalNodeID n0 = 3 * rank;

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

  DistributedGraph graph{};

private:
  GlobalNodeID next(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u + step) % n;
  }

  GlobalNodeID prev(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u < step) ? n + u - step : u - step;
  }
};

TEST(LocalContractionTest, ContractingEdgeWorks) {}
} // namespace dkaminpar::test