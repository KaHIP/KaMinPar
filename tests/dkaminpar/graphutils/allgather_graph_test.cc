#include <gmock/gmock.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/graphutils/allgather_graph.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

TEST(GraphReplicationTest, isolated_graph_1) {
    auto graph = make_isolated_nodes_graph(1);
    auto rep   = graph::replicate(graph, 1);

    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

    // Only 1 copy -> graph should stay the same
    EXPECT_EQ(rep.n(), 1);
    EXPECT_EQ(rep.global_n(), size);
    EXPECT_EQ(rep.m(), 0);
}
} // namespace kaminpar::dist
