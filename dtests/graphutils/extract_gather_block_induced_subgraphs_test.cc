#include <gmock/gmock.h>

#include "common/datastructures/static_array.h"
#include "dkaminpar/graphutils/graph_extraction.h"
#include "dtests/distributed_graph_fixtures.h"
#include "dtests/graph_assertions.h"
#include "dtests/graph_helpers.h"
#include "kaminpar/datastructure/graph.h"

using namespace dkaminpar;
using namespace dkaminpar::testing;
using namespace dkaminpar::testing::fixtures;

inline auto extract_subgraphs(const DistributedPartitionedGraph& p_graph) {
    return dkaminpar::graph::distribute_block_induced_subgraphs(p_graph).subgraphs;
}

// One isolated node on each PE, no edges at all
using OneIsolatedNodeOnEachPE = DistributedIsolatedNodesGraphFixture<1>;
TEST_F(OneIsolatedNodeOnEachPE, extracts_local_node) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = extract_subgraphs(p_graph);

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
    auto subgraphs = extract_subgraphs(p_graph);

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
    auto subgraphs = extract_subgraphs(p_graph);

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
    auto subgraphs = extract_subgraphs(p_graph);

    // each PE should get one block
    ASSERT_EQ(subgraphs.size(), 1);

    // the block should contain a single edge
    const auto& subgraph = subgraphs.front();
    ASSERT_EQ(subgraph.n(), 2);
    EXPECT_EQ(subgraph.degree(0), 1);
    EXPECT_EQ(subgraph.degree(1), 1);
    EXPECT_EQ(subgraph.m(), 2);
}

// Test with 10 local egdes
using TenEdgesOnEachPE = DistributedEdgesGraphFixture<10>;
TEST_F(TenEdgesOnEachPE, extracts_local_edges) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = extract_subgraphs(p_graph);

    // each PE should still get one block
    ASSERT_EQ(subgraphs.size(), 1);

    // the block should contain 10 edges
    const auto& subgraph = subgraphs.front();
    ASSERT_EQ(subgraph.n(), 20);
    EXPECT_EQ(subgraph.m(), 20);

    for (const NodeID u: subgraph.nodes()) {
        EXPECT_EQ(subgraph.degree(u), 1);
        const NodeID neighbor = subgraph.edge_target(subgraph.first_edge(u));
        EXPECT_EQ(subgraph.degree(neighbor), 1);
        const NodeID neighbor_neighbor = subgraph.edge_target(subgraph.first_edge(neighbor));
        EXPECT_EQ(neighbor_neighbor, u);
    }
}

// Test with cut edges: ring across PEs, but there should be still no local egdes
TEST_F(DistributedCircleGraphFixture, extracts_local_node) {
    auto p_graph   = make_partitioned_graph_by_rank(graph);
    auto subgraphs = extract_subgraphs(p_graph);

    // each PE should still get one block
    ASSERT_EQ(subgraphs.size(), 1);

    // each block should contain a single node
    const auto& subgraph = subgraphs.front();
    ASSERT_EQ(subgraph.n(), 1);
    EXPECT_EQ(subgraph.m(), 0);
}

// Test extracting isolated nodes that are spread across PEs
TEST_F(DistributedTestFixture, extracts_distributed_isolated_nodes) {
    // create graph with one local node for each PE
    auto                 graph = make_distributed_isolated_graph(size);
    std::vector<BlockID> partition(size);
    std::iota(partition.begin(), partition.end(), 0);
    auto p_graph = make_partitioned_graph(graph, static_cast<BlockID>(size), partition);

    auto subgraphs = extract_subgraphs(p_graph);

    // each PE should get one block
    ASSERT_EQ(subgraphs.size(), 1);
    const auto& subgraph = subgraphs.front();

    // ... with size isolated nodes
    ASSERT_EQ(subgraph.n(), size);
    ASSERT_EQ(subgraph.m(), 0);
}

void expect_circle(const shm::Graph& graph) {
    NodeID num_nodes_in_circle = 1;
    NodeID start               = 0;
    NodeID prev                = start;
    NodeID cur                 = graph.degree(start) > 0 ? graph.edge_target(graph.first_edge(start)) : start;

    while (cur != start) {
        EXPECT_EQ(graph.degree(cur), 2);

        const NodeID neighbor1 = graph.edge_target(graph.first_edge(cur));
        const NodeID neighbor2 = graph.edge_target(graph.first_edge(cur) + 1);
        EXPECT_TRUE(neighbor1 == prev || neighbor2 == prev);

        // move to next node
        prev = cur;
        cur  = (neighbor1 == prev) ? neighbor2 : neighbor1;

        ++num_nodes_in_circle;
    }

    EXPECT_EQ(num_nodes_in_circle, graph.n());
}

// Test local clique + global circle extraction, where nodes within a clique belong to different blocks
TEST_F(DistributedTestFixture, extract_circles_from_clique_graph) {
    auto                 graph = make_distributed_circle_clique_graph(size);
    std::vector<BlockID> partition(size);
    std::iota(partition.begin(), partition.end(), 0);
    auto p_graph = make_partitioned_graph(graph, static_cast<BlockID>(size), partition);

    auto subgraphs = extract_subgraphs(p_graph);

    // each PE should still get one block
    ASSERT_EQ(subgraphs.size(), 1);
    const auto& subgraph = subgraphs.front();

    // each block should be a circle
    ASSERT_EQ(subgraph.n(), size);

    if (size == 1) {
        EXPECT_EQ(subgraph.m(), 0);
    } else if (size == 2) {
        EXPECT_EQ(subgraph.m(), 2);
    } else {
        EXPECT_EQ(subgraph.m(), 2 * size);
    }

    expect_circle(subgraph);
}

// Test extracting two blocks per PE, each block with a isolated node
TEST_F(TwoIsolatedNodesOnEachPE, extract_two_isolated_node_blocks_per_pe) {
    auto p_graph =
        make_partitioned_graph(graph, 2 * size, {static_cast<BlockID>(2 * rank), static_cast<BlockID>(2 * rank + 1)});
    auto subgraphs = extract_subgraphs(p_graph);

    // two blocks per PE
    ASSERT_EQ(subgraphs.size(), 2);

    // each containing a single block
    for (const auto& subgraph: subgraphs) {
        EXPECT_EQ(subgraph.n(), 1);
        EXPECT_EQ(subgraph.m(), 0);
    }
}

// Test extracting two blocks, both containing a circle
TEST_F(DistributedTestFixture, extract_two_blocks_from_clique_graph) {
    auto                 graph = make_distributed_circle_clique_graph(2 * size); // two nodes per PE
    std::vector<BlockID> local_partition(2 * size);
    for (const NodeID u: graph.nodes()) {
        local_partition[u] = u;
    }
    auto p_graph = make_partitioned_graph(graph, 2 * size, local_partition);

    auto subgraphs = extract_subgraphs(p_graph);

    // two blocks per PE
    ASSERT_EQ(subgraphs.size(), 2);

    // each containing a circle
    for (const auto& subgraph: subgraphs) {
        EXPECT_EQ(subgraph.n(), size);

        if (size == 1) {
            EXPECT_EQ(subgraph.m(), 0);
        } else if (size == 2) {
            EXPECT_EQ(subgraph.m(), 2); // just two nodes with an edge between them
        } else {
            EXPECT_EQ(subgraph.m(), 2 * size);
        }

        expect_circle(subgraph);
    }
}

// Test node weights
TEST_F(DistributedTestFixture, node_weights_are_correct) {
    // create clique/circle graph with rank as node weight
    auto                                       graph = make_distributed_circle_clique_graph(2 * size);
    std::vector<std::pair<NodeID, NodeWeight>> node_weights;
    std::vector<BlockID>                       local_partition;
    for (const NodeID u: graph.nodes()) {
        node_weights.emplace_back(u, rank);
        local_partition.push_back(u);
    }
    graph          = change_node_weights(std::move(graph), node_weights);
    auto p_graph   = make_partitioned_graph(graph, 2 * size, local_partition);
    auto subgraphs = extract_subgraphs(p_graph);

    ASSERT_EQ(subgraphs.size(), 2);

    for (const auto& subgraph: subgraphs) {
        // we assume that the i-th node in a subgraph comes from PE i and thus has weight i
        // this should always be true with the current implementation, but might be too assertive?
        for (const NodeID u: subgraph.nodes()) {
            EXPECT_EQ(subgraph.node_weight(u), u);
        }
    }
}

// Test copying subgraph partition back to the distributed graph: one isolated nodes that is not migrated
TEST_F(OneIsolatedNodeOnEachPE, copy_partition_back_to_distributed_graph) {
    auto p_graph = make_partitioned_graph_by_rank(graph);

    auto  result    = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);
    auto& subgraphs = result.subgraphs;

    // one block with one node -> assign to block 0
    auto& subgraph = subgraphs.front();

    std::vector<shm::PartitionedGraph> p_subgraphs;
    shm::StaticArray<BlockID>          partition(1);
    partition[0] = 0;
    p_subgraphs.push_back({subgraph, 1, std::move(partition), {1}});

    // Copy back to p_graph
    p_graph = dkaminpar::graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

    EXPECT_EQ(p_graph.k(), size); // k should not have changed
    ASSERT_EQ(p_graph.n(), 1);
    EXPECT_EQ(p_graph.block(0), rank); // partition should not have changed
}

// ... test with two nodes
TEST_F(TwoIsolatedNodesOnEachPE, copy_partition_back_to_distributed_graph) {
    auto p_graph = make_partitioned_graph_by_rank(graph);

    auto  result    = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);
    auto& subgraphs = result.subgraphs;

    // one block with one node -> assign to block 0
    auto& subgraph = subgraphs.front();

    std::vector<shm::PartitionedGraph> p_subgraphs;
    shm::StaticArray<BlockID>          partition(2);
    partition[0] = 0;
    partition[1] = 1;
    p_subgraphs.push_back({subgraph, 2, std::move(partition), {1, 1}});

    // Copy back to p_graph
    p_graph = dkaminpar::graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

    EXPECT_EQ(p_graph.k(), 2 * size); // k should not have doubled
    ASSERT_EQ(p_graph.n(), 2);
    // We cannot tell which node is in which block, only that one should be in block 0 and one in block 1
    EXPECT_NE(p_graph.block(0), p_graph.block(1));
    EXPECT_EQ(p_graph.block(0) + p_graph.block(1), 2 * rank + 2 * rank + 1);
}

// ... test with clique
TEST_F(DistributedTestFixture, copy_partition_back_to_distributed_graph_circle) {
    auto graph = make_distributed_circle_clique_graph(2 * size); // two nodes per PE

    // Always place two nodes in one partition
    std::vector<BlockID> local_partition(2 * size);
    for (const NodeID u: graph.nodes()) {
        local_partition[u] = u / 2;
    }
    auto p_graph = make_partitioned_graph(graph, size, local_partition);

    // Extract blocks
    auto  result    = dkaminpar::graph::distribute_block_induced_subgraphs(p_graph);
    auto& subgraphs = result.subgraphs;
    ASSERT_EQ(subgraphs.size(), 1);
    auto& subgraph = subgraphs.front();

    // Should have 2 * size nodes on each PE
    ASSERT_EQ(subgraph.n(), 2 * size);

    // Assign 2 nodes to a new block
    std::vector<shm::PartitionedGraph> p_subgraphs;
    shm::StaticArray<BlockID>          partition(2 * size);
    for (const NodeID u: subgraph.nodes()) {
        partition[u] = u / 2;
    }
    p_subgraphs.push_back(
        {subgraph, static_cast<BlockID>(size), std::move(partition), scalable_vector<BlockID>(size / 2, 1)});

    // Copy back to p_graph
    p_graph = dkaminpar::graph::copy_subgraph_partitions(std::move(p_graph), p_subgraphs, result);

    // Should have size * (size / 2) blocks now
    ASSERT_EQ(p_graph.k(), size * size);

    for (const NodeID u: p_graph.nodes()) {
        EXPECT_EQ(p_graph.block(u), (u / 2) * size + rank);
    }
}
