/*******************************************************************************
 * @file:   shm_parhip_io_test.cc
 * @brief:  Unit tests for shared-memory ParHiP graph IO.
 ******************************************************************************/
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kaminpar-io/graph_compression_binary.h"
#include "kaminpar-io/kaminpar_io.h"
#include "kaminpar-io/parhip_parser.h"
#include "tests/io/shm_io_helpers.h"

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

namespace kaminpar::shm::testing {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Eq;
using namespace io_helpers;

std::uint64_t parhip_version(const bool has_node_weights, const bool has_edge_weights) {
  const auto make_flag = [](const bool flag, const std::uint64_t shift) {
    return static_cast<std::uint64_t>(flag ? 0 : 1) << shift;
  };
  return make_flag(sizeof(EdgeWeight) == 8, 5) | make_flag(sizeof(NodeWeight) == 8, 4) |
         make_flag(sizeof(NodeID) == 8, 3) | make_flag(sizeof(EdgeID) == 8, 2) |
         make_flag(has_node_weights, 1) | make_flag(has_edge_weights, 0);
}

void write_direct_parhip(const std::string &filename, const GraphSnapshot &snapshot) {
  std::ofstream out(filename, std::ios::binary);
  write_binary<std::uint64_t>(out, parhip_version(snapshot.node_weighted, snapshot.edge_weighted));
  write_binary<std::uint64_t>(out, snapshot.n);
  write_binary<std::uint64_t>(out, snapshot.m);

  const EdgeID nodes_offset_base = 3 * sizeof(std::uint64_t) + (snapshot.n + 1) * sizeof(EdgeID);
  EdgeID cur_edge = 0;
  for (NodeID u = 0; u < snapshot.n; ++u) {
    write_binary<EdgeID>(out, nodes_offset_base + cur_edge * sizeof(NodeID));
    cur_edge += static_cast<EdgeID>(snapshot.adjacency[u].size());
  }
  write_binary<EdgeID>(out, nodes_offset_base + cur_edge * sizeof(NodeID));

  for (const auto &adjacency : snapshot.adjacency) {
    for (const Edge edge : adjacency) {
      write_binary<NodeID>(out, edge.target);
    }
  }
  if (snapshot.node_weighted) {
    for (const NodeWeight weight : snapshot.node_weights) {
      write_binary<NodeWeight>(out, weight);
    }
  }
  if (snapshot.edge_weighted) {
    for (const auto &adjacency : snapshot.adjacency) {
      for (const Edge edge : adjacency) {
        write_binary<EdgeWeight>(out, edge.weight);
      }
    }
  }
}

void expect_parhip_rejected(const std::string &filename) {
  for (const bool compress : {false, true}) {
    for (const NodeOrdering ordering :
         {NodeOrdering::NATURAL, NodeOrdering::EXTERNAL_DEGREE_BUCKETS}) {
      SCOPED_TRACE(compress);
      EXPECT_FALSE(io::parhip::read_graph(filename, compress, ordering));
    }
  }
}

} // namespace

TEST(ShmParhipIOTest, direct_fixtures_read_as_csr_and_compressed) {
  const std::vector<GraphSnapshot> snapshots = {
      rgg16_snapshot(false, false),
      rgg16_snapshot(true, false),
      rgg16_snapshot(false, true),
      rgg16_snapshot(true, true),
  };

  for (const GraphSnapshot &expected : snapshots) {
    TempFile file(".parhip");
    write_direct_parhip(file.string(), expected);

    for (const bool compress : {false, true}) {
      for (const NodeOrdering ordering :
           {NodeOrdering::NATURAL, NodeOrdering::IMPLICIT_DEGREE_BUCKETS}) {
        SCOPED_TRACE(compress);
        const auto graph = io::parhip::read_graph(file.string(), compress, ordering);
        ASSERT_TRUE(graph);
        EXPECT_EQ(graph->is_compressed(), compress);
        EXPECT_EQ(graph->sorted(), ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS);
        expect_graph_eq(*graph, expected);
      }
    }
  }
}

TEST(ShmParhipIOTest, external_degree_bucket_read_preserves_weight_identified_graph) {
  const GraphSnapshot expected = weighted_test_snapshot();
  TempFile file(".parhip");
  write_direct_parhip(file.string(), expected);

  for (const bool compress : {false, true}) {
    SCOPED_TRACE(compress);
    const auto graph =
        io::parhip::read_graph(file.string(), compress, NodeOrdering::EXTERNAL_DEGREE_BUCKETS);
    ASSERT_TRUE(graph);
    EXPECT_EQ(graph->is_compressed(), compress);
    EXPECT_TRUE(graph->sorted());
    expect_graph_eq_by_unique_node_weight(*graph, expected);
  }
}

TEST(ShmParhipIOTest, write_graph_roundtrips_and_uses_expected_header) {
  const GraphSnapshot expected = weighted_test_snapshot();
  const Graph graph = make_graph(expected);
  TempFile file(".parhip");

  io::parhip::write_graph(file.string(), graph);

  std::ifstream in(file.string(), std::ios::binary);
  std::uint64_t version;
  std::uint64_t n;
  std::uint64_t m;
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&m), sizeof(m));
  EXPECT_EQ(version, parhip_version(true, true));
  EXPECT_EQ(n, expected.n);
  EXPECT_EQ(m, expected.m);

  const EdgeID nodes_offset_base = 3 * sizeof(std::uint64_t) + (expected.n + 1) * sizeof(EdgeID);
  EdgeID cur_edge = 0;
  for (NodeID u = 0; u < expected.n; ++u) {
    EdgeID raw_node;
    in.read(reinterpret_cast<char *>(&raw_node), sizeof(raw_node));
    EXPECT_EQ(raw_node, nodes_offset_base + cur_edge * sizeof(NodeID));
    cur_edge += static_cast<EdgeID>(expected.adjacency[u].size());
  }
  EdgeID raw_sentinel;
  in.read(reinterpret_cast<char *>(&raw_sentinel), sizeof(raw_sentinel));
  EXPECT_EQ(raw_sentinel, nodes_offset_base + expected.m * sizeof(NodeID));

  std::vector<NodeID> raw_edges(expected.m);
  in.read(reinterpret_cast<char *>(raw_edges.data()), raw_edges.size() * sizeof(NodeID));
  EXPECT_THAT(raw_edges, ElementsAreArray(flattened_edges(expected)));

  const auto read = io::read_graph(file.string(), io::GraphFileFormat::PARHIP);
  ASSERT_TRUE(read);
  expect_graph_eq(*read, expected);

  const auto compressed =
      io::read_graph(file.string(), io::GraphFileFormat::PARHIP, true, NodeOrdering::NATURAL);
  ASSERT_TRUE(compressed);
  EXPECT_TRUE(compressed->is_compressed());
  expect_graph_eq(*compressed, expected);
}

TEST(ShmParhipIOTest, parhip_reader_rejects_malformed_files) {
  {
    TempFile file(".parhip");
    std::ofstream out(file.string(), std::ios::binary);
    write_binary<std::uint64_t>(out, parhip_version(false, false));
    out.close();
    expect_parhip_rejected(file.string());
  }

  {
    TempFile file(".parhip");
    write_direct_parhip(file.string(), rgg16_snapshot(false, false));
    std::ofstream out(file.string(), std::ios::binary | std::ios::app);
    write_binary<std::uint8_t>(out, 0xff);
    out.close();
    expect_parhip_rejected(file.string());
  }

  {
    TempFile file(".parhip");
    std::ofstream out(file.string(), std::ios::binary);
    write_binary<std::uint64_t>(out, parhip_version(false, false));
    write_binary<std::uint64_t>(out, 2);
    write_binary<std::uint64_t>(out, 2);
    const EdgeID nodes_offset_base = 3 * sizeof(std::uint64_t) + 3 * sizeof(EdgeID);
    write_binary<EdgeID>(out, nodes_offset_base);
    write_binary<EdgeID>(out, nodes_offset_base + 2 * sizeof(NodeID));
    write_binary<EdgeID>(out, nodes_offset_base + sizeof(NodeID));
    write_binary<NodeID>(out, 1);
    write_binary<NodeID>(out, 0);
    out.close();
    expect_parhip_rejected(file.string());
  }

  {
    TempFile file(".parhip");
    std::ofstream out(file.string(), std::ios::binary);
    write_binary<std::uint64_t>(out, parhip_version(false, false));
    write_binary<std::uint64_t>(out, 2);
    write_binary<std::uint64_t>(out, 2);
    const EdgeID nodes_offset_base = 3 * sizeof(std::uint64_t) + 3 * sizeof(EdgeID);
    write_binary<EdgeID>(out, nodes_offset_base);
    write_binary<EdgeID>(out, nodes_offset_base + sizeof(NodeID));
    write_binary<EdgeID>(out, nodes_offset_base + 2 * sizeof(NodeID));
    write_binary<NodeID>(out, 1);
    write_binary<NodeID>(out, 2);
    out.close();
    expect_parhip_rejected(file.string());
  }
}

TEST(ShmParhipIOTest, compressed_binary_roundtrips_compressed_graphs) {
  const GraphSnapshot expected = weighted_test_snapshot();
  Graph graph = make_graph(expected);
  Graph compressed = parallel_compress(graph.csr_graph());
  TempFile file(".compressed");

  io::write_graph(file.string(), io::GraphFileFormat::COMPRESSED, compressed);
  const auto read = io::read_graph(file.string(), io::GraphFileFormat::COMPRESSED);

  ASSERT_TRUE(read);
  EXPECT_TRUE(read->is_compressed());
  expect_graph_eq(*read, expected);
}

TEST(ShmParhipIOTest, compressed_binary_rejects_missing_file_and_wrong_magic) {
  TempFile missing(".compressed");
  EXPECT_FALSE(io::read_graph(missing.string(), io::GraphFileFormat::COMPRESSED));

  TempFile wrong_magic(".compressed");
  {
    std::ofstream out(wrong_magic.string(), std::ios::binary);
    write_binary<std::uint64_t>(out, 0x123456789abcdef0ull);
  }
  EXPECT_FALSE(io::read_graph(wrong_magic.string(), io::GraphFileFormat::COMPRESSED));

  TempFile truncated(".compressed");
  {
    std::ofstream out(truncated.string(), std::ios::binary);
    write_binary<std::uint64_t>(out, io::compressed_binary::kMagicNumber);
  }
  EXPECT_FALSE(io::read_graph(truncated.string(), io::GraphFileFormat::COMPRESSED));

  TempFile trailing(".compressed");
  Graph graph = make_graph(weighted_test_snapshot());
  Graph compressed = parallel_compress(graph.csr_graph());
  io::write_graph(trailing.string(), io::GraphFileFormat::COMPRESSED, compressed);
  {
    std::ofstream out(trailing.string(), std::ios::binary | std::ios::app);
    write_binary<std::uint8_t>(out, 0xff);
  }
  EXPECT_FALSE(io::read_graph(trailing.string(), io::GraphFileFormat::COMPRESSED));
}

TEST(ShmParhipIOTest, partition_and_block_size_helpers_roundtrip) {
  TempFile partition_file(".partition");
  const std::vector<BlockID> partition = {2, 0, 2, 1, 1, 0, 2};
  io::write_partition(partition_file.string(), partition);
  EXPECT_THAT(io::read_partition(partition_file.string()), ElementsAreArray(partition));

  TempFile block_sizes_file(".block-sizes");
  io::write_block_sizes(block_sizes_file.string(), 3, partition);
  EXPECT_THAT(
      io::read_block_sizes(block_sizes_file.string()),
      ElementsAreArray(std::vector<BlockID>{0, 0, 1, 1, 2, 2, 2})
  );
}

TEST(ShmParhipIOTest, write_remapping_writes_one_node_per_line) {
  TempFile file(".map");
  const std::vector<NodeID> remapping = {4, 2, 0, 3, 1};
  io::write_remapping(file.string(), remapping);
  EXPECT_THAT(read_file(file.string()), Eq("4\n2\n0\n3\n1\n"));
}

} // namespace kaminpar::shm::testing
