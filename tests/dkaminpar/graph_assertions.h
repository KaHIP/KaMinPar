#include <gmock/gmock.h>

#include "dkaminpar/datastructure/distributed_graph.h"

#include "kaminpar/datastructure/graph.h"

namespace dkaminpar::testing {
inline void expect_triangle_graph(const shm::Graph& graph) {
    EXPECT_EQ(graph.n(), 3);
    EXPECT_EQ(graph.m(), 6);

    for (const NodeID u: graph.nodes()) {
        EXPECT_EQ(graph.degree(u), 2);

        // expect two distinct neighbors
        const NodeID neighbor_1 = graph.edge_target(graph.first_edge(u));
        const NodeID neighbor_2 = graph.edge_target(graph.first_edge(u) + 1);
        EXPECT_NE(neighbor_1, neighbor_2);
        EXPECT_NE(u, neighbor_1);
        EXPECT_NE(u, neighbor_2);
        EXPECT_LT(neighbor_1, graph.n());
        EXPECT_LT(neighbor_2, graph.n());
    }
}

inline void expect_unweighted_graph(
    const shm::Graph& graph, const NodeWeight expected_node_weight = 1, const EdgeWeight expected_edge_weight = 1) {
    for (const NodeID u: graph.nodes()) {
        EXPECT_EQ(graph.node_weight(u), expected_node_weight);
    }
    for (const EdgeID e: graph.edges()) {
        EXPECT_EQ(graph.edge_weight(e), expected_edge_weight);
    }
}
} // namespace dkaminpar::testing
