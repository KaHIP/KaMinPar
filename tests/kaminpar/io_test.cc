#include <gmock/gmock.h>

#include "tests/basic_test_helpers.h"
#include "tests/kaminpar/test_helpers.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/io.h"

using ::testing::UnorderedElementsAre;
using namespace kaminpar::testing;
using namespace kaminpar::shm::testing;

namespace kaminpar::shm {
inline void assert_K3_structure(const Graph& G) {
    EXPECT_EQ(G.n(), 3);
    EXPECT_EQ(G.m(), 6);
    EXPECT_THAT(view_to_vector(G.adjacent_nodes(0)), UnorderedElementsAre(1, 2));
    EXPECT_THAT(view_to_vector(G.adjacent_nodes(1)), UnorderedElementsAre(0, 2));
    EXPECT_THAT(view_to_vector(G.adjacent_nodes(2)), UnorderedElementsAre(0, 1));
}

inline auto outgoing_edge_weights(const Graph& G, const NodeID u) {
    std::vector<EdgeWeight> edge_weights;
    for (const EdgeID& e: G.incident_edges(u)) {
        edge_weights.push_back(G.edge_weight(e));
    }
    return edge_weights;
}

TEST(IOTest, unweighted_K3) {
    const auto G = io::metis::read<false>(test_instance("unweighted_K3.graph"));
    assert_K3_structure(G);

    for (const NodeID& u: G.nodes()) {
        EXPECT_EQ(G.node_weight(u), 1);
    }
    for (const EdgeID& e: G.edges()) {
        EXPECT_EQ(G.edge_weight(e), 1);
    }
}

TEST(IOTest, node_weighted_K3) {
    const auto G = io::metis::read<false>(test_instance("node_weighted_K3.graph"));
    assert_K3_structure(G);

    EXPECT_EQ(G.node_weight(0), 1);
    EXPECT_EQ(G.node_weight(1), 2);
    EXPECT_EQ(G.node_weight(2), 3);
    for (const EdgeID& e: G.edges()) {
        EXPECT_EQ(G.edge_weight(e), 1);
    }
}

TEST(IOTest, edge_weighted_K3) {
    const auto G = io::metis::read<false>(test_instance("edge_weighted_K3.graph"));
    assert_K3_structure(G);

    EXPECT_THAT(outgoing_edge_weights(G, 0), UnorderedElementsAre(1, 2));
    EXPECT_THAT(outgoing_edge_weights(G, 1), UnorderedElementsAre(1, 3));
    EXPECT_THAT(outgoing_edge_weights(G, 2), UnorderedElementsAre(2, 3));
}

TEST(IOTest, weighted_K3) {
    const auto G = io::metis::read<false>(test_instance("weighted_K3.graph"));
    assert_K3_structure(G);

    EXPECT_EQ(G.node_weight(0), 10);
    EXPECT_EQ(G.node_weight(1), 20);
    EXPECT_EQ(G.node_weight(2), 30);

    EXPECT_THAT(outgoing_edge_weights(G, 0), UnorderedElementsAre(1, 2));
    EXPECT_THAT(outgoing_edge_weights(G, 1), UnorderedElementsAre(1, 3));
    EXPECT_THAT(outgoing_edge_weights(G, 2), UnorderedElementsAre(2, 3));
}

TEST(IOTest, large_weights) {
    const auto G = io::metis::read<false>(test_instance("large_weights.graph"));

    EXPECT_EQ(G.node_weight(0), 123456789);
    EXPECT_EQ(G.node_weight(1), 234567891);
}

TEST(IOTest, graph_with_comments) {
    const auto G = io::metis::read<false>(test_instance("with_comments.graph"));

    EXPECT_EQ(G.n(), 2);
    EXPECT_EQ(G.m(), 2);
    EXPECT_THAT(view_to_vector(G.adjacent_nodes(0)), UnorderedElementsAre(1));
    EXPECT_THAT(view_to_vector(G.adjacent_nodes(1)), UnorderedElementsAre(0));
}
} // namespace kaminpar::shm
