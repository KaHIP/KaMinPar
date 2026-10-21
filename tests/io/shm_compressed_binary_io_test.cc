/*******************************************************************************
 * @file:   shm_compressed_binary_io_test.cc
 ******************************************************************************/
#include <fstream>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "kaminpar-io/graph_compression_binary.h"
#include "tests/io/shm_io_helpers.h"

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

namespace kaminpar::shm::testing {

namespace {

using namespace io_helpers;

GraphSnapshot streamvbyte_test_snapshot() {
  GraphSnapshot snapshot;
  snapshot.n = 100;
  snapshot.m = 4 * snapshot.n;
  snapshot.total_node_weight = snapshot.n;
  snapshot.total_edge_weight = snapshot.m;
  snapshot.node_weights.assign(snapshot.n, 1);
  snapshot.adjacency.resize(snapshot.n);

  // Four nonconsecutive neighbors exercise StreamVByte without forming an interval.
  for (NodeID u = 0; u < snapshot.n; ++u) {
    for (const NodeID offset : {1, 7, 93, 99}) {
      snapshot.adjacency[u].push_back({(u + offset) % snapshot.n, 1});
    }
    std::sort(snapshot.adjacency[u].begin(), snapshot.adjacency[u].end());
  }
  return snapshot;
}

} // namespace

TEST(ShmCompressedBinaryIOTest, roundtrips_serial_and_parallel_compression_with_decoder_padding) {
  const GraphSnapshot expected = streamvbyte_test_snapshot();
  const Graph graph = make_graph(expected);

  for (const bool parallel : {false, true}) {
    SCOPED_TRACE(parallel);
    Graph compressed =
        parallel ? parallel_compress(graph.csr_graph()) : compress(graph.csr_graph());
    const auto &concrete_graph = compressed.compressed_graph();
    constexpr std::size_t kPadding = CompressedGraph::kStreamVByteEncoding ? 15 : 0;
    EXPECT_EQ(
        concrete_graph.raw_nodes()[concrete_graph.n()] + kPadding,
        concrete_graph.raw_compressed_edges().size()
    );

    TempFile file(".compressed");
    io::compressed_binary::write(file.string(), concrete_graph);
    const auto read = io::compressed_binary::read(file.string());
    ASSERT_TRUE(read);
    expect_graph_eq(*read, expected);
  }
}

TEST(ShmCompressedBinaryIOTest, rejects_terminal_offsets_before_or_after_encoded_data) {
  const Graph graph = make_graph(streamvbyte_test_snapshot());

  for (const int delta : {-1, 1}) {
    SCOPED_TRACE(delta);
    Graph compressed = compress(graph.csr_graph());
    auto &concrete_graph = compressed.compressed_graph();
    auto &nodes = concrete_graph.raw_nodes();
    const EdgeID terminal = nodes[concrete_graph.n()];
    nodes.write(concrete_graph.n(), delta < 0 ? terminal - 1 : terminal + 1);

    TempFile file(".compressed");
    io::compressed_binary::write(file.string(), concrete_graph);
    EXPECT_FALSE(io::compressed_binary::read(file.string()));
  }
}

TEST(ShmCompressedBinaryIOTest, rejects_nonmonotone_offsets_across_parallel_validation_chunks) {
  constexpr NodeID kNumNodes = (1u << 20) + 1;
  CompactStaticArray<EdgeID> nodes(1, kNumNodes + 1);
  for (NodeID u = 0; u <= kNumNodes; ++u) {
    nodes.write(u, 0);
  }
  nodes.write(1u << 18, 1);

  constexpr std::size_t kPadding = CompressedGraph::kStreamVByteEncoding ? 15 : 0;
  StaticArray<std::uint8_t> compressed_edges(kPadding, std::uint8_t{0});
  CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight> neighborhoods(
      std::move(nodes), std::move(compressed_edges), 0, 0, false, 0, 0, 0, 0, 0
  );
  const CompressedGraph graph(std::move(neighborhoods), {}, false);
  TempFile file(".compressed");
  io::compressed_binary::write(file.string(), graph);
  EXPECT_FALSE(io::compressed_binary::read(file.string()));
}

TEST(ShmCompressedBinaryIOTest, rejects_bytes_after_the_serialized_arrays) {
  const Graph graph = make_graph(streamvbyte_test_snapshot());
  const Graph compressed = compress(graph.csr_graph());
  TempFile file(".compressed");
  io::compressed_binary::write(file.string(), compressed.compressed_graph());
  {
    std::ofstream out(file.string(), std::ios::binary | std::ios::app);
    out.put('\0');
  }
  EXPECT_FALSE(io::compressed_binary::read(file.string()));
}

} // namespace kaminpar::shm::testing
