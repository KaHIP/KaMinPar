#include <gmock/gmock.h>

#include "dtests/distributed_graph_fixtures.h"
#include "dtests/graph_assertions.h"
#include "dtests/graph_helpers.h"

#include "dkaminpar/graphutils/graph_extraction.h"

using namespace dkaminpar;
using namespace dkaminpar::testing;
using namespace dkaminpar::testing::fixtures;

// One isolated node on each PE, no edges at all
using OneIsolatedNodeOnEachPE = DistributedIsolatedNodesGraphFixture<1>;
TEST_F(OneIsolatedNodeOnEachPE, extracts_local_node) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);

    // each PE should get one block
    ASSERT_EQ(subgraphs.size(), 1);

    // ech block should be a single node without any neighbors
    const auto& subgraph = subgraphs.front();
    EXPECT_EQ(subgraph.n(), 1);
    EXPECT_EQ(subgraph.m(), 0);
    EXPECT_EQ(subgraph.total_node_weight(), 1);
    EXPECT_EQ(subgraph.total_edge_weight(), 0);
}

// Two isolated nodes on each PE, no edges at all
using TwoIsolatedNodesOnEachPE = DistributedIsolatedNodesGraphFixture<2>;
TEST_F(TwoIsolatedNodesOnEachPE, extracts_local_nodes) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);

    // each PE should get one block
    ASSERT_EQ(subgraphs.size(), 1);

    // each block should consist of two isolated nodes
    const auto& subgraph = subgraphs.front();
    EXPECT_EQ(subgraph.n(), 2);
    EXPECT_EQ(subgraph.m(), 0);
    EXPECT_EQ(subgraph.total_node_weight(), 2);
    EXPECT_EQ(subgraph.total_edge_weight(), 0);
}

// Test empty blocks
TEST_F(EmptyGraphFixture, extracts_empty_graphs) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);

    // still expect one (empty) block pe PE
    ASSERT_EQ(subgraphs.size(), 1);

    const auto& subgraph = subgraphs.front();
    EXPECT_EQ(subgraph.n(), 0);
    EXPECT_EQ(subgraph.m(), 0);
    EXPECT_EQ(subgraph.total_node_weight(), 0);
    EXPECT_EQ(subgraph.total_edge_weight(), 0);
}

// Test with local egdes
using OneEdgeOnEachPE = DistributedEdgesGraphFixture<1>;
TEST_F(OneEdgeOnEachPE, extracts_local_edge) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);

    // each PE should get one block 
    ASSERT_EQ(subgraphs.size(), 1);

    // the block should contain a single edge 
    const auto &subgraph = subgraphs.front();
    ASSERT_EQ(subgraph.n(), 2);
    EXPECT_EQ(subgraph.degree(0), 1);
    EXPECT_EQ(subgraph.degree(1), 1);
    EXPECT_EQ(subgraph.m(), 2);
}
