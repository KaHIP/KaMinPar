#include <gmock/gmock.h>

#include "dtests/distributed_graph_fixtures.h"
#include "dtests/graph_assertions.h"
#include "dtests/graph_helpers.h"

#include "dkaminpar/graphutils/graph_extraction.h"

using namespace dkaminpar;
using namespace dkaminpar::testing;
using namespace dkaminpar::testing::fixtures;

TEST_F(DistributedIsolatedNodesGraphFixture, extracts_local_node) {
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
