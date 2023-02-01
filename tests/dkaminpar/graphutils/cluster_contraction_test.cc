/*******************************************************************************
 * @file:   contraction_test.cc
 * @author: Daniel Seemaier
 * @date:   28.01.2023
 * @brief:  Unit tests for the graph contraction function.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/mpi/utils.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

TEST(ClusterContractionTest, contract_empty_graph) {
    auto graph = make_empty_graph();

    GlobalClustering clustering;
    auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.n(), 0);
    EXPECT_EQ(c_graph.global_n(), 0);
    EXPECT_EQ(c_graph.m(), 0);
    EXPECT_EQ(c_graph.global_m(), 0);
}

TEST(ClusterContractionTest, contract_local_edge) {
    const auto graph = make_isolated_edges_graph(1);
    const PEID size  = mpi::get_comm_size(MPI_COMM_WORLD);

    GlobalClustering clustering{0, 0};
    auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), size);
    EXPECT_EQ(c_graph.m(), 0);
    EXPECT_EQ(c_graph.global_m(), 0);
    ASSERT_EQ(c_graph.n(), 1);
    EXPECT_EQ(c_graph.node_weight(0), 2);
}
} // namespace kaminpar::dist
