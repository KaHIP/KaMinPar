/*******************************************************************************
 * @file:   rearrange_graph_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   18.11.21
 * @brief:  Unit tests for rearranging distributed graphs by node degree.
 ******************************************************************************/
#include "dtests/mpi_test.h"

#include "dkaminpar/graphutils/rearrange_graph.h"

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Eq;

namespace dkaminpar::test {
using namespace fixtures3PE;

TEST_F(UnsortedDistributedPath, RearrangingUnsortedPathWorks) {
    // +-------------+
    // +------+      |
    // 0--1-#-2--3-#-4--5
    //        +------+
    EXPECT_EQ(graph.total_node_weight(), 2);

    graph = ::dkaminpar::graph::sort_by_degree_buckets(std::move(graph));

    EXPECT_EQ(graph.total_node_weight(), 2);
    for (const NodeID u: graph.all_nodes()) {
        EXPECT_EQ(graph.node_weight(u), 1);
    }

    EXPECT_EQ(graph.degree(0), 1);
    EXPECT_EQ(graph.degree(1), 3);

    for (auto [e, v]: graph.neighbors(0)) { // deg 1 node
        EXPECT_TRUE(graph.is_owned_node(v));
        EXPECT_EQ(v, 1);
        EXPECT_EQ(graph.local_to_global_node(v), n0 + 1);
    }

    for (auto [e, v]: graph.neighbors(1)) { // deg 3 node
        if (graph.is_owned_node(v)) {
            EXPECT_EQ(v, 0);
            EXPECT_EQ(graph.local_to_global_node(v), n0);
        } else {
            EXPECT_THAT(graph.local_to_global_node(v), AnyOf(1, 3, 5));
        }
    }
}
} // namespace dkaminpar::test