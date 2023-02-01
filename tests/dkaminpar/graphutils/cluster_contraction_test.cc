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

    GlobalClustering clustering{graph.offset_n(), graph.offset_n()};
    auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), size);
    EXPECT_EQ(c_graph.m(), 0);
    EXPECT_EQ(c_graph.global_m(), 0);
    ASSERT_EQ(c_graph.n(), 1);
    EXPECT_EQ(c_graph.node_weight(0), 2);
}

TEST(ClusterContractionTest, contract_local_clique) {
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

    for (const NodeID clique_size: {1, 5, 10}) {
        const auto graph = make_local_clique_graph(clique_size);

        GlobalClustering clustering(clique_size, graph.offset_n());
        auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

        EXPECT_EQ(c_graph.global_n(), size);
        EXPECT_EQ(c_graph.global_m(), 0);
        EXPECT_EQ(c_graph.m(), 0);
        ASSERT_EQ(c_graph.n(), 1);
        EXPECT_EQ(c_graph.node_weight(0), static_cast<NodeWeight>(clique_size));
    }
}

TEST(ClusterContractionTest, contract_local_complete_bipartite_graph_vertically) {
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

    for (const NodeID set_size: {1, 5, 10}) {
        const auto graph = make_local_complete_bipartite_graph(set_size);

        GlobalClustering clustering(2 * set_size);
        std::fill(clustering.begin(), clustering.begin() + set_size, graph.offset_n());
        std::fill(clustering.begin() + set_size, clustering.end(), graph.offset_n() + set_size);
        auto [c_graph, c_mapping] = contract_clustering(graph, clustering);

        EXPECT_EQ(c_graph.global_n(), 2 * size);
        EXPECT_EQ(c_graph.global_m(), 2 * size);

        ASSERT_EQ(c_graph.n(), 2);
        EXPECT_EQ(c_graph.node_weight(0), set_size);
        EXPECT_EQ(c_graph.node_weight(1), set_size);

        ASSERT_EQ(c_graph.m(), 2);
        EXPECT_EQ(c_graph.edge_weight(0), set_size * set_size);
        EXPECT_EQ(c_graph.edge_weight(1), set_size * set_size);
    }
}
} // namespace kaminpar::dist
