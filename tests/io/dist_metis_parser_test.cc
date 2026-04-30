/*******************************************************************************
 * @file:   dist_metis_parser_test.cc
 * @brief:  Unit tests for distributed METIS graph parsing.
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-io/dist_metis_parser.h"

#include "kaminpar-mpi/utils.h"

namespace kaminpar::dist::io::metis {

namespace {

struct MetisFixture {
  const char *filename;
  bool has_node_weights;
  bool has_edge_weights;
  GlobalNodeWeight total_node_weight;
  GlobalEdgeWeight total_edge_weight;
};

constexpr std::size_t kNumNodesSize = 16;
constexpr GlobalNodeID kNumNodes = static_cast<GlobalNodeID>(kNumNodesSize);
constexpr GlobalEdgeID kNumEdges = 2 * 24;
constexpr GlobalNodeWeight kWeightedNodeWeight = 136;
constexpr GlobalEdgeWeight kWeightedEdgeWeight = 2 * 332;

constexpr MetisFixture kMetisFixtures[] = {
    {
        .filename = "tests/io/rgg16.metis",
        .has_node_weights = false,
        .has_edge_weights = false,
        .total_node_weight = kNumNodes,
        .total_edge_weight = kNumEdges,
    },
    {
        .filename = "tests/io/rgg16-vwgt.metis",
        .has_node_weights = true,
        .has_edge_weights = false,
        .total_node_weight = kWeightedNodeWeight,
        .total_edge_weight = kNumEdges,
    },
    {
        .filename = "tests/io/rgg16-adjwgt.metis",
        .has_node_weights = false,
        .has_edge_weights = true,
        .total_node_weight = kNumNodes,
        .total_edge_weight = kWeightedEdgeWeight,
    },
    {
        .filename = "tests/io/rgg16-vwgt-adjwgt.metis",
        .has_node_weights = true,
        .has_edge_weights = true,
        .total_node_weight = kWeightedNodeWeight,
        .total_edge_weight = kWeightedEdgeWeight,
    },
};

using ExpectedAdjacency = std::vector<std::pair<GlobalNodeID, EdgeWeight>>;

const std::array<ExpectedAdjacency, kNumNodesSize> kWeightedTopology = {
    ExpectedAdjacency{{1, 1}, {8, 21}, {15, 16}},
    ExpectedAdjacency{{0, 1}, {2, 2}, {9, 22}},
    ExpectedAdjacency{{1, 2}, {3, 3}, {10, 23}},
    ExpectedAdjacency{{2, 3}, {4, 4}, {11, 24}},
    ExpectedAdjacency{{3, 4}, {5, 5}, {12, 25}},
    ExpectedAdjacency{{4, 5}, {6, 6}, {13, 26}},
    ExpectedAdjacency{{5, 6}, {7, 7}, {14, 27}},
    ExpectedAdjacency{{6, 7}, {8, 8}, {15, 28}},
    ExpectedAdjacency{{0, 21}, {7, 8}, {9, 9}},
    ExpectedAdjacency{{1, 22}, {8, 9}, {10, 10}},
    ExpectedAdjacency{{2, 23}, {9, 10}, {11, 11}},
    ExpectedAdjacency{{3, 24}, {10, 11}, {12, 12}},
    ExpectedAdjacency{{4, 25}, {11, 12}, {13, 13}},
    ExpectedAdjacency{{5, 26}, {12, 13}, {14, 14}},
    ExpectedAdjacency{{6, 27}, {13, 14}, {15, 15}},
    ExpectedAdjacency{{0, 16}, {7, 28}, {14, 15}},
};

ExpectedAdjacency expected_adjacency(const GlobalNodeID global_u, const MetisFixture &fixture) {
  auto expected = kWeightedTopology[static_cast<std::size_t>(global_u)];
  if (!fixture.has_edge_weights) {
    for (auto &edge : expected) {
      edge.second = 1;
    }
  }
  std::sort(expected.begin(), expected.end());
  return expected;
}

NodeWeight expected_node_weight(const GlobalNodeID global_u, const MetisFixture &fixture) {
  return fixture.has_node_weights ? static_cast<NodeWeight>(global_u + 1) : 1;
}

template <typename Graph>
void expect_fixture_graph(const Graph &graph, const MetisFixture &fixture) {
  const auto rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  EXPECT_EQ(kNumNodes, graph.global_n());
  EXPECT_EQ(kNumEdges, graph.global_m());

  EXPECT_EQ(graph.node_distribution(rank + 1) - graph.node_distribution(rank), graph.n());
  EXPECT_EQ(graph.edge_distribution(rank + 1) - graph.edge_distribution(rank), graph.m());

  EXPECT_EQ(fixture.has_node_weights, graph.is_node_weighted());
  EXPECT_EQ(fixture.has_edge_weights, graph.is_edge_weighted());
  EXPECT_EQ(fixture.total_node_weight, graph.global_total_node_weight());
  EXPECT_EQ(fixture.total_edge_weight, graph.global_total_edge_weight());

  if (fixture.has_node_weights) {
    EXPECT_EQ(graph.total_n(), graph.node_weights().size());
  } else {
    EXPECT_TRUE(graph.node_weights().empty());
  }

  for (const NodeID u : graph.nodes()) {
    const GlobalNodeID global_u = graph.local_to_global_node(u);
    const ExpectedAdjacency expected = expected_adjacency(global_u, fixture);

    EXPECT_EQ(expected_node_weight(global_u, fixture), graph.node_weight(u))
        << "global node " << global_u;
    EXPECT_EQ(static_cast<NodeID>(expected.size()), graph.degree(u)) << "global node " << global_u;

    ExpectedAdjacency actual;
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      actual.emplace_back(graph.local_to_global_node(v), weight);
    });
    std::sort(actual.begin(), actual.end());

    EXPECT_THAT(actual, ::testing::ElementsAreArray(expected)) << "global node " << global_u;
  }
}

} // namespace

TEST(DistMetisParserTest, compressed_read_reads_metis_fixtures) {
  for (const MetisFixture &fixture : kMetisFixtures) {
    SCOPED_TRACE(fixture.filename);
    const auto graph =
        compress_read(fixture.filename, GraphDistribution::BALANCED_NODES, false, MPI_COMM_WORLD);
    expect_fixture_graph(graph, fixture);
  }
}

TEST(DistMetisParserTest, csr_read_reads_metis_fixtures) {
  for (const MetisFixture &fixture : kMetisFixtures) {
    SCOPED_TRACE(fixture.filename);
    const auto graph =
        csr_read(fixture.filename, GraphDistribution::BALANCED_NODES, false, MPI_COMM_WORLD);
    expect_fixture_graph(graph, fixture);
  }
}

} // namespace kaminpar::dist::io::metis
