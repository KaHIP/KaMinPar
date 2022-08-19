/*******************************************************************************
 * @file:   extract_subgraphs_test.cc
 * @author: Daniel Seemaier
 * @date:   29.04.2022
 * @brief:  Unit tests to test the extraction of block induced subgraphs.
 ******************************************************************************/
#include "tests/dkaminpar/mpi_test.h"

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/graphutils/graph_extraction.h"

using namespace testing;
using namespace dkaminpar::test::fixtures3PE;

namespace kaminpar::dist {
TEST_F(UnsortedDistributedPath, ExtractSubgraphsWithOneBlockOnEachPE) {
    // +-------------+
    // +------+      |
    // 0--1 # 2--3 # 4--5
    //        +------+
    const BlockID b       = static_cast<BlockID>(rank);
    auto          p_graph = test::graph::make_partitioned_graph(graph, 3, {b, b});

    auto result = graph::extract_local_block_induced_subgraphs(p_graph);

    EXPECT_EQ(result.shared_nodes.size(), 2);
    EXPECT_THAT(result.shared_nodes, ElementsAre(1, 2));

    EXPECT_EQ(result.shared_edges.size(), 2);
    EXPECT_THAT(result.shared_edges, ElementsAre(1, 0));

    EXPECT_EQ(result.shared_node_weights.size(), 2);
    EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1));

    EXPECT_EQ(result.shared_edge_weights.size(), 2);
    EXPECT_THAT(result.shared_edge_weights, ElementsAre(1, 1));

    switch (rank) {
        case 0:
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 2, 2, 2));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 2, 2, 2));
            break;
        case 1:
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 0, 2, 2));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 2, 2));
            break;
        case 2:
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 0, 0, 2));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 0, 2));
            break;
    }
}

TEST_F(DistributedTriangles, ExtractSubgraphsWithRotated3rdNodes) {
    //  0---1-#-3---4
    //  |\ /  #  \ /|
    //  | 2---#---5 |
    //  |  \  #  /  |
    // ###############
    //  |    \ /    |
    //  |     8     |
    //  |    / \    |
    //  +---7---6---+
    const BlockID b       = static_cast<BlockID>(rank);
    const auto    p_graph = test::graph::make_partitioned_graph(graph, 3, {b, b, (b + 1) % size});
    const auto    result  = graph::extract_local_block_induced_subgraphs(p_graph);

    EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1, 1));
    EXPECT_THAT(result.shared_edge_weights, ElementsAre(1, 1));

    switch (rank) {
        case 0:
            EXPECT_THAT(result.shared_nodes, ElementsAre(1, 2, 0));
            EXPECT_THAT(result.shared_edges, ElementsAre(1, 0));
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 2, 3, 3));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 2, 2, 2));
            break;

        case 1:
            EXPECT_THAT(result.shared_nodes, ElementsAre(1, 2, 0));
            EXPECT_THAT(result.shared_edges, ElementsAre(2, 1));
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 0, 2, 3));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 2, 2));
            break;

        case 2:
            EXPECT_THAT(result.shared_nodes, ElementsAre(0, 1, 2));
            EXPECT_THAT(result.shared_edges, ElementsAre(2, 1));
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 1, 1, 3));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 0, 2));
            break;
    }
}

TEST_F(DistributedTriangles, ExtractSubgraphsWithOneBlockPerPE) {
    //  0---1-#-3---4
    //  |\ /  #  \ /|
    //  | 2---#---5 |
    //  |  \  #  /  |
    // ###############
    //  |    \ /    |
    //  |     8     |
    //  |    / \    |
    //  +---7---6---+
    const BlockID b       = static_cast<BlockID>(rank);
    const auto    p_graph = test::graph::make_partitioned_graph(graph, 3, {b, b, b});
    const auto    result  = graph::extract_local_block_induced_subgraphs(p_graph);

    EXPECT_THAT(result.shared_nodes, ElementsAre(2, 4, 6));
    EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1, 1));
    EXPECT_THAT(result.shared_edge_weights, ElementsAre(1, 1, 1, 1, 1, 1));
    EXPECT_THAT(result.shared_edges, UnorderedElementsAre(1, 2, 0, 2, 0, 1));

    switch (rank) {
        case 0:
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 3, 3, 3));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 6, 6, 6));
            break;

        case 1:
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 0, 3, 3));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 6, 6));
            break;

        case 2:
            EXPECT_THAT(result.nodes_offset, ElementsAre(0, 0, 0, 3));
            EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 0, 6));
            break;
    }
}

TEST_F(DistributedTriangles, ExtractSubgraphsWithOneNodePerSubgraph) {
    //  0---1-#-3---4
    //  |\ /  #  \ /|
    //  | 2---#---5 |
    //  |  \  #  /  |
    // ###############
    //  |    \ /    |
    //  |     8     |
    //  |    / \    |
    //  +---7---6---+
    auto p_graph = test::graph::make_partitioned_graph(graph, 3, {0, 1, 2});
    auto result  = graph::extract_local_block_induced_subgraphs(p_graph);

    // DLOG << V(result.shared_nodes) << V(result.shared_edges) << V(result.shared_node_weights)
    //      << V(result.shared_edge_weights) << V(result.nodes_offset) << V(result.edges_offset);

    EXPECT_THAT(result.shared_nodes, ElementsAre(0, 0, 2));
    EXPECT_THAT(result.shared_node_weights, ElementsAre(1, 1, 1));
    EXPECT_THAT(result.shared_edge_weights, ElementsAre(1, 1));
    EXPECT_THAT(result.nodes_offset, ElementsAre(0, 1, 2, 3));
    EXPECT_THAT(result.edges_offset, ElementsAre(0, 0, 0, 2));

    switch (rank) {
        case 0:
            EXPECT_THAT(result.shared_edges, UnorderedElementsAre(1, 2));
            break;

        case 1:
            EXPECT_THAT(result.shared_edges, UnorderedElementsAre(0, 2));
            break;

        case 2:
            EXPECT_THAT(result.shared_edges, UnorderedElementsAre(0, 1));
            break;
    }
}
} // namespace kaminpar::dist
