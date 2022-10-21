#include <gmock/gmock.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/graphutils/allgather_graph.h"

#include "common/assert.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

TEST(GraphReplicationTest, isolated_graph_1) {
    const auto graph = make_isolated_nodes_graph(1);
    const auto rep   = graph::replicate(graph, 1);
    const PEID size  = mpi::get_comm_size(MPI_COMM_WORLD);

    // Only 1 copy -> graph should stay the same
    EXPECT_EQ(rep.n(), 1);
    EXPECT_EQ(rep.global_n(), size);
    EXPECT_EQ(rep.m(), 0);
}

TEST(GraphReplicationTest, isolated_graph_P) {
    const auto graph = make_isolated_nodes_graph(1);
    const PEID size  = mpi::get_comm_size(MPI_COMM_WORLD);
    const auto rep   = graph::replicate(graph, size);

    // size copies -> every PE should own the full graph
    EXPECT_EQ(rep.n(), size);
    EXPECT_EQ(rep.global_n(), size);
    EXPECT_EQ(rep.m(), 0);
}

TEST(GraphReplicationTest, isolated_graph_P_div_2) {
    const auto graph = make_isolated_nodes_graph(1);
    const PEID size  = mpi::get_comm_size(MPI_COMM_WORLD);

    if (size > 2) {
        KASSERT(size % 2 == 0, "unit tests only works if number of PEs is divisable by 2", assert::always);
        const auto rep = graph::replicate(graph, size / 2);

        EXPECT_EQ(rep.n(), 2);
        EXPECT_EQ(rep.global_n(), size);
        EXPECT_EQ(rep.m(), 0);
    }
}

TEST(GraphReplicationTest, triangle_cycle_graph_1) {
    const auto graph = make_circle_clique_graph(3); // triangle on each PE
    const auto rep   = graph::replicate(graph, 1);  // replicate 1 == nothing changes

    EXPECT_EQ(rep.n(), graph.n());
    EXPECT_EQ(rep.global_n(), graph.global_n());
    EXPECT_EQ(rep.m(), graph.m());
    EXPECT_EQ(rep.global_m(), graph.global_m());

    for (const NodeID u: graph.nodes()) {
        EXPECT_EQ(rep.degree(u), graph.degree(u));
    }
}

TEST(GraphReplicationTest, triangle_cycle_graph_P) {
    const auto graph = make_circle_clique_graph(3); // triangle on each PE
    const PEID size  = mpi::get_comm_size(MPI_COMM_WORLD);
    const auto rep   = graph::replicate(graph, size); // each PE gets a full copy

    EXPECT_EQ(rep.n(), rep.global_n());
    EXPECT_EQ(rep.n(), size * 3);
    EXPECT_EQ(rep.m(), rep.global_m());
}
} // namespace kaminpar::dist
