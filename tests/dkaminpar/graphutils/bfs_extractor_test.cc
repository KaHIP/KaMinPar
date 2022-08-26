/*******************************************************************************
 * @file:   bfs_extractor_test.cc
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 * @brief:  Unit tests for BFS subgraph extraction.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/graphutils/bfs_extractor.h"

#include "kaminpar/datastructure/graph.h"

namespace kaminpar::dist::graph {
using namespace kaminpar::dist::testing;

std::vector<shm::Graph>
extract_bfs_subgraphs(DistributedPartitionedGraph& p_graph, const std::vector<NodeID>& seed_nodes) {
    BfsExtractor extractor(p_graph);

    std::vector<shm::Graph> bfs_graphs;
    auto                    results = extractor.extract(seed_nodes);
    for (auto& result: results) {
        bfs_graphs.push_back(std::move(result.graph()));
    }
    return bfs_graphs;
}

shm::Graph extract_bfs_subgraph(DistributedPartitionedGraph& p_graph, const NodeID seed_node) {
    return std::move(extract_bfs_subgraphs(p_graph, {seed_node}).front());
}

TEST(BfsExtractor, empty_graph) {
    auto graph      = make_empty_graph();
    auto p_graph    = make_partitioned_graph_by_rank(graph);
    auto bfs_graphs = extract_bfs_subgraphs(p_graph, {});
    EXPECT_TRUE(bfs_graphs.empty());
}
} // namespace kaminpar::dist::graph

