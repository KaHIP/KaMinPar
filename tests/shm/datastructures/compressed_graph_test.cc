#include <bitset>
#include <map>

#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"

#include "kaminpar-common/variable_length_codec.h"

#define TEST_ON_ALL_GRAPHS(test_function)                                                          \
  test_function(graphs::empty(0));                                                                 \
  test_function(graphs::empty(100));                                                               \
  test_function(graphs::path(100));                                                                \
  test_function(graphs::star(100));                                                                \
  test_function(graphs::grid(100, 100));                                                           \
  test_function(graphs::complete_bipartite(100, 100));                                             \
  test_function(graphs::complete(100));                                                            \
  test_function(graphs::matching(100));

namespace kaminpar::shm::testing {

template <typename T> static bool operator==(const IotaRange<T> &a, const IotaRange<T> &b) {
  return a.begin() == b.begin() && a.end() == b.end();
};

static void print_compressed_graph(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

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

  std::cout << '\n';
}

static void test_graph_compression(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

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

TEST(CompressedGraphTest, compression) {
  TEST_ON_ALL_GRAPHS(test_graph_compression);
}

static void test_compressed_graph_nodes_operation(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

  EXPECT_TRUE(csr_graph.nodes() == compressed_graph.nodes());
}

TEST(CompressedGraphTest, compressed_graph_nodes_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_nodes_operation);
}

static void test_compressed_graph_edges_operation(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

  EXPECT_TRUE(csr_graph.edges() == compressed_graph.edges());
}

TEST(CompressedGraphTest, compressed_graph_edges_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_edges_operation);
}

static void test_compressed_graph_degree_operation(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

  for (const NodeID node : graph.nodes()) {
    EXPECT_EQ(csr_graph.degree(node), compressed_graph.degree(node));
  }
}

TEST(CompressedGraphTest, compressed_graph_degree_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_degree_operation);
}

static void test_compressed_graph_incident_edges_operation(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

  for (const NodeID node : graph.nodes()) {
    EXPECT_TRUE(csr_graph.incident_edges(node) == compressed_graph.incident_edges(node));
  }
}

TEST(CompressedGraphTest, compressed_graph_incident_edges_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_incident_edges_operation);
}

static void test_compressed_graph_neighbors_operation(const Graph &graph) {
  const CSRGraph &csr_graph = *dynamic_cast<const CSRGraph *>(graph.underlying_graph());
  const auto compressed_graph = CompressedGraph::compress(csr_graph);

  for (const NodeID node : graph.nodes()) {
    std::vector<EdgeID> graph_incident_edges;
    std::vector<NodeID> graph_adjacent_node;
    std::vector<EdgeID> compressed_graph_incident_edges;
    std::vector<NodeID> compressed_graph_adjacent_node;

    for (const auto [incident_edge, adjacent_node] : graph.neighbors(node)) {
      graph_incident_edges.push_back(incident_edge);
      graph_adjacent_node.push_back(adjacent_node);
    }

    for (const auto [incident_edge, adjacent_node] : compressed_graph.neighbors(node)) {
      compressed_graph_incident_edges.push_back(incident_edge);
      compressed_graph_adjacent_node.push_back(adjacent_node);
    }

    EXPECT_EQ(graph_incident_edges.size(), compressed_graph_incident_edges.size());

    std::sort(graph_incident_edges.begin(), graph_incident_edges.end());
    std::sort(graph_adjacent_node.begin(), graph_adjacent_node.end());
    std::sort(compressed_graph_incident_edges.begin(), compressed_graph_incident_edges.end());
    std::sort(compressed_graph_adjacent_node.begin(), compressed_graph_adjacent_node.end());
    EXPECT_TRUE(graph_incident_edges == compressed_graph_incident_edges);
    EXPECT_TRUE(graph_adjacent_node == compressed_graph_adjacent_node);
  }
}

TEST(CompressedGraphTest, compressed_graph_neighbors_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_neighbors_operation);
}

} // namespace kaminpar::shm::testing