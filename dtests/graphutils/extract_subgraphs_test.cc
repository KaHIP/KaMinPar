/*******************************************************************************
 * @file:   extract_subgraphs_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.04.2022
 * @brief:  Unit tests to test the extraction of block induced subgraphs.
 ******************************************************************************/
#include "datastructure/distributed_graph.h"
#include "dkaminpar/graphutils/graph_extraction.h"
#include "dtests/mpi_test.h"

using namespace testing;
using namespace dkaminpar::test::fixtures3PE;

namespace dkaminpar {
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
} // namespace dkaminpar
