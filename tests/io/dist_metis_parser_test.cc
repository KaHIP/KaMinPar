/*******************************************************************************
 * @file:   dist_metis_parser_test.cc
 * @brief:  Unit tests for distributed METIS graph parsing.
 ******************************************************************************/
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

constexpr GlobalNodeID kNumNodes = 16;
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

template <typename Graph> void expect_fixture_graph(const Graph &graph, const MetisFixture &fixture) {
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
