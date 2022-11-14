/***********************************************************************************************************************
 * @file:   greedy_node_coloring_test.cc
 * @author: Daniel Seemaier
 * @date:   11.11.2022
 * @brief:  Unit tests for the greedy node (vertex) coloring algorithm.
 **********************************************************************************************************************/
#include <limits>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/mpi/utils.h"
#include "dkaminpar/mpi/wrapper.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;

namespace {
template <typename Coloring>
void validate_node_coloring(
    const DistributedGraph& graph, const Coloring& coloring,
    const ColorID max_num_colors = std::numeric_limits<ColorID>::max()
) {
    ASSERT_GE(coloring.size(), graph.total_n());
    for (const NodeID u: graph.nodes()) {
        EXPECT_LT(coloring[u], max_num_colors);
        for (const NodeID v: graph.adjacent_nodes(u)) {
            EXPECT_NE(coloring[u], coloring[v]);
        }
    }
}
} // namespace

TEST(GreedyNodeColoringTest, colors_empty_graph) {
    auto graph    = make_empty_graph();
    auto coloring = compute_node_coloring_sequentially(graph, 1);
    EXPECT_TRUE(coloring.empty());
}

TEST(GreedyNodeColoringTest, colors_isolated_nodes_graph) {
    constexpr NodeID kNodesPerPE         = 2;
    constexpr NodeID kNumberOfSupersteps = 1;

    const auto graph    = make_isolated_nodes_graph(kNodesPerPE);
    const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
    validate_node_coloring(graph, coloring, 1);
}

TEST(GreedyNodeColoringTest, colors_isolated_edges_graph) {
    constexpr NodeID kNodesPerPE         = 2;
    constexpr NodeID kNumberOfSupersteps = 1;

    const auto graph    = make_isolated_edges_graph(kNodesPerPE);
    const auto coloring = compute_node_coloring_sequentially(graph, kNumberOfSupersteps);
    validate_node_coloring(graph, coloring, 2);
}

/*
TEST(IndependentBorderSetTest, select_in_circle_graph) {
    auto graph   = make_circle_graph();
    auto p_graph = make_partitioned_graph_by_rank(graph);

    auto is = find_independent_border_set(p_graph, 0);

    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    if (size == 1) { // No border on just one PE
        ASSERT_TRUE(is.empty());
    } else {
        expect_nonempty_independent_set(p_graph, is);
    }
}

TEST(IndependentBorderSetTest, select_in_cut_edge_graph_10) {
    for (const NodeID num_nodes_per_pe: {1, 2, 8, 16, 32, 64}) {
        auto graph     = make_cut_edge_graph(num_nodes_per_pe);
        auto p_graph   = make_partitioned_graph_by_rank(graph);
        auto is        = find_independent_border_set(p_graph, 0);
        auto global_is = allgather_independent_set(p_graph, is);

        const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
        if (size == 1) {
            ASSERT_TRUE(global_is.empty());
        } else {
            ASSERT_EQ(global_is.size(), size * num_nodes_per_pe);
            expect_nonempty_independent_set(p_graph, is);
        }
    }
}

TEST(IndependentBorderSetTest, randomization_works) {
    auto graph   = make_cut_edge_graph(1000);
    auto p_graph = make_partitioned_graph_by_rank(graph);

    auto a_is        = find_independent_border_set(p_graph, 0);
    auto a_global_is = allgather_independent_set(p_graph, a_is);

    auto b_is        = find_independent_border_set(p_graph, 1);
    auto b_global_is = allgather_independent_set(p_graph, b_is);

    // prob. for same IS negetible
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    if (size == 1) {
        EXPECT_EQ(a_global_is, b_global_is);
    } else {
        EXPECT_NE(a_global_is, b_global_is);
    }
}
*/
} // namespace kaminpar::dist
