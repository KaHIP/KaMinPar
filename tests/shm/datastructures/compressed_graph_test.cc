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

template <typename CompressedGraph> static void test_graph_compression(const Graph &graph) {
  const auto compressed_graph = CompressedGraph::compress(graph);

  EXPECT_EQ(graph.n(), compressed_graph.n());
  EXPECT_EQ(graph.m(), compressed_graph.m());

  for (const NodeID node : graph.nodes()) {
    std::vector<NodeID> graph_neighbours;
    std::vector<NodeID> compressed_graph_neighbours;

    for (const NodeID adjacent_node : graph.adjacent_nodes(node)) {
      graph_neighbours.push_back(adjacent_node);
    }

    for (const NodeID adjacent_node : compressed_graph.adjacent_nodes(node)) {
      compressed_graph_neighbours.push_back(adjacent_node);
    }

    EXPECT_EQ(graph_neighbours.size(), compressed_graph_neighbours.size());

    std::sort(graph_neighbours.begin(), graph_neighbours.end());
    std::sort(compressed_graph_neighbours.begin(), compressed_graph_neighbours.end());
    EXPECT_TRUE(graph_neighbours == compressed_graph_neighbours);
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
  using CompressedGraph = CompressedGraph<VarIntCodec, false>;
  test_graph_compression<CompressedGraph>();
}

TEST(CompressedGraphTest, interval_encoding) {
  using CompressedGraph = CompressedGraph<VarIntCodec, true>;
  test_graph_compression<CompressedGraph>();
}

} // namespace kaminpar::shm::testing