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

std::pair<std::unique_ptr<shm::Graph>, std::unique_ptr<shm::PartitionedGraph>>
extract_bfs_subgraph(DistributedPartitionedGraph& p_graph, const std::vector<NodeID>& seed_nodes) {
    BfsExtractor extractor(p_graph.graph());
    extractor.initialize(p_graph);
    auto result = extractor.extract(seed_nodes);
    return {std::move(result.graph), std::move(result.p_graph)};
}

TEST(BfsExtractor, empty_graph) {
    auto graph                    = make_empty_graph();
    auto p_graph                  = make_partitioned_graph_by_rank(graph);
    auto [bfs_graph, p_bfs_graph] = extract_bfs_subgraph(p_graph, {});
    EXPECT_EQ(bfs_graph->n(), 0);
}
} // namespace kaminpar::dist::graph

