/*******************************************************************************
 * @file:   dist_parhip_parser_test.cc
 * @brief:  Unit tests for distributed ParHIP graph parsing.
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <unistd.h>

#include "kaminpar-io/dist_parhip_parser.h"

#include "kaminpar-mpi/utils.h"

namespace kaminpar::dist::io::parhip {

namespace {

struct Edge {
  GlobalNodeID target;
  EdgeWeight weight;
};

using ExpectedAdjacency = std::vector<std::pair<GlobalNodeID, EdgeWeight>>;

constexpr GlobalNodeID kNumNodes = 4;
constexpr GlobalEdgeID kNumEdges = 8;
constexpr GlobalNodeWeight kTotalNodeWeight = 10;
constexpr GlobalEdgeWeight kTotalEdgeWeight = 96;

const std::array<NodeWeight, kNumNodes> kNodeWeights = {1, 2, 3, 4};
const std::array<std::vector<Edge>, kNumNodes> kTopology = {
    std::vector<Edge>{{1, 7}, {2, 11}},
    std::vector<Edge>{{0, 7}, {3, 13}},
    std::vector<Edge>{{0, 11}, {3, 17}},
    std::vector<Edge>{{1, 13}, {2, 17}},
};

template <typename T> void write_binary(std::ofstream &out, const T value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

std::uint64_t parhip_version(
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool has_64_bit_edge_id,
    const bool has_64_bit_node_id,
    const bool has_64_bit_node_weight,
    const bool has_64_bit_edge_weight
) {
  const auto make_flag = [](const bool flag, const std::uint64_t shift) {
    return static_cast<std::uint64_t>(flag ? 0 : 1) << shift;
  };
  return make_flag(has_64_bit_edge_weight, 5) | make_flag(has_64_bit_node_weight, 4) |
         make_flag(has_64_bit_node_id, 3) | make_flag(has_64_bit_edge_id, 2) |
         make_flag(has_node_weights, 1) | make_flag(has_edge_weights, 0);
}

template <
    typename FileEdgeID,
    typename FileNodeID,
    typename FileNodeWeight,
    typename FileEdgeWeight>
void write_parhip_as(const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  write_binary<std::uint64_t>(
      out,
      parhip_version(
          true,
          true,
          sizeof(FileEdgeID) == 8,
          sizeof(FileNodeID) == 8,
          sizeof(FileNodeWeight) == 8,
          sizeof(FileEdgeWeight) == 8
      )
  );
  write_binary<std::uint64_t>(out, kNumNodes);
  write_binary<std::uint64_t>(out, kNumEdges);

  const std::uint64_t nodes_offset_base =
      3 * sizeof(std::uint64_t) + (kNumNodes + 1) * sizeof(FileEdgeID);
  std::uint64_t cur_edge = 0;
  for (GlobalNodeID u = 0; u < kNumNodes; ++u) {
    write_binary<FileEdgeID>(
        out, static_cast<FileEdgeID>(nodes_offset_base + cur_edge * sizeof(FileNodeID))
    );
    cur_edge += kTopology[u].size();
  }
  write_binary<FileEdgeID>(
      out, static_cast<FileEdgeID>(nodes_offset_base + cur_edge * sizeof(FileNodeID))
  );

  for (const auto &adjacency : kTopology) {
    for (const Edge edge : adjacency) {
      write_binary<FileNodeID>(out, static_cast<FileNodeID>(edge.target));
    }
  }
  for (const NodeWeight weight : kNodeWeights) {
    write_binary<FileNodeWeight>(out, static_cast<FileNodeWeight>(weight));
  }
  for (const auto &adjacency : kTopology) {
    for (const Edge edge : adjacency) {
      write_binary<FileEdgeWeight>(out, static_cast<FileEdgeWeight>(edge.weight));
    }
  }
}

std::string make_temp_filename(const char *suffix) {
  std::string filename;
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    filename = (std::filesystem::temp_directory_path() /
                ("kaminpar-dist-parhip-test-" + std::to_string(::getpid()) + suffix))
                   .string();
  }

  int length = static_cast<int>(filename.size());
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  filename.resize(length);
  MPI_Bcast(filename.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  return filename;
}

ExpectedAdjacency expected_adjacency(const GlobalNodeID global_u) {
  ExpectedAdjacency expected;
  for (const Edge edge : kTopology[static_cast<std::size_t>(global_u)]) {
    expected.emplace_back(edge.target, edge.weight);
  }
  std::sort(expected.begin(), expected.end());
  return expected;
}

template <typename Graph> void expect_graph(const Graph &graph) {
  const auto rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  EXPECT_EQ(kNumNodes, graph.global_n());
  EXPECT_EQ(kNumEdges, graph.global_m());
  EXPECT_EQ(graph.node_distribution(rank + 1) - graph.node_distribution(rank), graph.n());
  EXPECT_EQ(graph.edge_distribution(rank + 1) - graph.edge_distribution(rank), graph.m());
  EXPECT_TRUE(graph.is_node_weighted());
  EXPECT_TRUE(graph.is_edge_weighted());
  EXPECT_EQ(kTotalNodeWeight, graph.global_total_node_weight());
  EXPECT_EQ(kTotalEdgeWeight, graph.global_total_edge_weight());

  for (const NodeID u : graph.nodes()) {
    const GlobalNodeID global_u = graph.local_to_global_node(u);
    EXPECT_EQ(kNodeWeights[static_cast<std::size_t>(global_u)], graph.node_weight(u));
    EXPECT_EQ(kTopology[static_cast<std::size_t>(global_u)].size(), graph.degree(u));

    ExpectedAdjacency actual;
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      actual.emplace_back(graph.local_to_global_node(v), weight);
    });
    std::sort(actual.begin(), actual.end());

    EXPECT_THAT(actual, ::testing::ElementsAreArray(expected_adjacency(global_u)))
        << "global node " << global_u;
  }
}

template <
    typename FileEdgeID,
    typename FileNodeID,
    typename FileNodeWeight,
    typename FileEdgeWeight>
void expect_width_variant(const char *suffix) {
  const std::string filename = make_temp_filename(suffix);
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    write_parhip_as<FileEdgeID, FileNodeID, FileNodeWeight, FileEdgeWeight>(filename);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for (const GraphDistribution distribution :
       {GraphDistribution::BALANCED_NODES, GraphDistribution::BALANCED_MEMORY_SPACE}) {
    SCOPED_TRACE(static_cast<int>(distribution));
    expect_graph(csr_read(filename, distribution, false, MPI_COMM_WORLD));
    expect_graph(compressed_read(filename, distribution, false, MPI_COMM_WORLD));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    std::filesystem::remove(filename);
  }
}

} // namespace

TEST(DistParhipParserTest, reads_integer_width_variants) {
  expect_width_variant<std::uint32_t, std::uint32_t, std::int32_t, std::int32_t>("-32.parhip");
  expect_width_variant<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t>("-64.parhip");
  expect_width_variant<std::uint64_t, std::uint32_t, std::int64_t, std::int32_t>("-mixed.parhip");
}

} // namespace kaminpar::dist::io::parhip
