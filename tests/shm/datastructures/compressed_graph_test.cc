#include <bitset>

#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"

#include "kaminpar-common/variable_length_codec.h"

namespace kaminpar::shm::testing {

template <typename CompressedGraph> static void print_compressed_graph(const Graph &graph) {
  const auto compressed_graph = CompressedGraph::compress(graph);
  const auto &nodes = compressed_graph.raw_nodes();
  const auto &compressed_edges = compressed_graph.raw_compressed_edges();

  std::cout << "Nodes: " << nodes.size() << ", edges: " << compressed_edges.size() << "\n\n";
  for (NodeID node = 0; node < nodes.size() - 1; ++node) {
    std::cout << "Node: " << node << ", offset: " << nodes[node] << '\n';

    const std::uint8_t *start = compressed_edges.data() + nodes[node];
    const std::uint8_t *end = compressed_edges.data() + nodes[node + 1];

    while (start < end) {
      std::cout << std::bitset<8>(*start++) << ' ';
    }
    std::cout << '\n';
  }
}

template <typename VarLenCodec, typename Int> static void test_varlen_codec(Int value) {
  std::size_t len = VarLenCodec::length(value);
  auto ptr = std::make_unique<std::uint8_t>(len);

  std::size_t encoded_value_len = VarLenCodec::encode(value, ptr.get());
  auto [decoded_value, decoded_value_len] = VarLenCodec::template decode<Int>(ptr.get());

  EXPECT_EQ(value, decoded_value);
  EXPECT_EQ(len, encoded_value_len);
  EXPECT_EQ(len, decoded_value_len);
}

template <typename VarLenCodec, typename Int> static void test_signed_varlen_codec(Int value) {
  std::size_t len = VarLenCodec::length_signed(value);
  auto ptr = std::make_unique<std::uint8_t>(len);

  std::size_t encoded_value_len = VarLenCodec::encode_signed(value, ptr.get());
  auto [decoded_value, decoded_value_len] = VarLenCodec::template decode_signed<Int>(ptr.get());

  EXPECT_EQ(value, decoded_value);
  EXPECT_EQ(len, encoded_value_len);
  EXPECT_EQ(len, decoded_value_len);
}

TEST(CompressedGraphTest, varlen_codec) {
  test_varlen_codec<VarIntCodec>(0);
  test_varlen_codec<VarIntCodec>(std::numeric_limits<std::size_t>::max() - 1);
}

TEST(CompressedGraphTest, varlen_codec_signed) {
  test_signed_varlen_codec<VarIntCodec>(0);
  test_signed_varlen_codec<VarIntCodec>(std::numeric_limits<int>::min() + 1);
  test_signed_varlen_codec<VarIntCodec>(std::numeric_limits<int>::max() - 1);
}

template <typename CompressedGraph> static void test_graph_compression(const Graph &graph) {
  const auto compressed_graph = CompressedGraph::compress(graph);

  EXPECT_EQ(graph.n(), compressed_graph.n());
  EXPECT_EQ(graph.m(), compressed_graph.m());

  for (const NodeID node : graph.nodes()) {
    const auto nodes = compressed_graph.adjacent_nodes(node);
    EXPECT_EQ(graph.degree(node), nodes.size());

    for (const NodeID adjacent_node : graph.adjacent_nodes(node)) {
      EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), adjacent_node) != nodes.end());
    }
  }
}

template <typename CompressedGraph> static void test_graph_compression() {
  test_graph_compression<CompressedGraph>(graphs::empty(0));
  test_graph_compression<CompressedGraph>(graphs::empty(100));
  test_graph_compression<CompressedGraph>(graphs::path(100));
  test_graph_compression<CompressedGraph>(graphs::star(100));
  test_graph_compression<CompressedGraph>(graphs::grid(100, 100));
  test_graph_compression<CompressedGraph>(graphs::complete_bipartite(100, 100));
  test_graph_compression<CompressedGraph>(graphs::complete(100));
  test_graph_compression<CompressedGraph>(graphs::matching(100));
}

TEST(CompressedGraphTest, gap_encoding) {
  using CompressedGraph = CompressedGraph<VarIntCodec, false, false>;
  test_graph_compression<CompressedGraph>();
}

TEST(CompressedGraphTest, interval_encoding) {
  using CompressedGraph = CompressedGraph<VarIntCodec, true, false>;
  test_graph_compression<CompressedGraph>();
}

TEST(CompressedGraphTest, reference_encoding) {
  using CompressedGraph = CompressedGraph<VarIntCodec, false, true>;
  test_graph_compression<CompressedGraph>();
}

TEST(CompressedGraphTest, reference_interval_encoding) {
  using CompressedGraph = CompressedGraph<VarIntCodec, true, true>;
  test_graph_compression<CompressedGraph>();
}

} // namespace kaminpar::shm::testing