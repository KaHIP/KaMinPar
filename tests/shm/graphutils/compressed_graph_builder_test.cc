#include <gmock/gmock.h>

#include <span>

#include <kaminpar-common/datastructures/static_array.h>
#include <kaminpar-shm/datastructures/csr_graph.h>
#include <kaminpar-shm/graphutils/compressed_graph_builder.h>

#include "tests/shm/graph_factories.h"

#define HIGH_DEGREE_NUM (CompressedGraph::kHighDegreeThreshold * 5)
#define TEST_ON_ALL_GRAPHS(test_function)                                                          \
  test_function(make_empty_graph(0));                                                              \
  test_function(make_empty_graph(100));                                                            \
  test_function(make_path_graph(100));                                                             \
  test_function(make_star_graph(100));                                                             \
  test_function(make_grid_graph(100, 100));                                                        \
  test_function(make_complete_bipartite_graph(100, 100));                                          \
  test_function(make_complete_graph(100));                                                         \
  test_function(make_matching_graph(100));                                                         \
  test_function(make_star_graph(HIGH_DEGREE_NUM));                                                 \
  test_function(make_complete_graph(100, [](const NodeID u, const NodeID v) {                      \
    return static_cast<EdgeWeight>(u + v);                                                         \
  }));                                                                                             \
  test_function(make_complete_bipartite_graph(100, 100, [](const NodeID u, const NodeID v) {       \
    return static_cast<EdgeWeight>(u + v);                                                         \
  }));                                                                                             \
  test_function(make_star_graph(HIGH_DEGREE_NUM, [](const NodeID u, const NodeID v) {              \
    return static_cast<EdgeWeight>(u + v);                                                         \
  }));

#define TEST_ON_ALL_GRAPHS2(test_function) test_function(make_complete_graph(5));

namespace kaminpar::shm::testing {

template <typename T> [[nodiscard]] std::span<T> to_span(StaticArray<T> &array) {
  return {array.data(), array.size()};
}

void is_correctly_compressed(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  ASSERT_EQ(graph.n(), compressed_graph.n());
  ASSERT_EQ(graph.m(), compressed_graph.m());
  ASSERT_EQ(graph.is_node_weighted(), compressed_graph.is_node_weighted());
  ASSERT_EQ(graph.is_edge_weighted(), compressed_graph.is_edge_weighted());
  ASSERT_EQ(graph.sorted(), compressed_graph.sorted());

  std::vector<NodeID> graph_neighbours;
  std::vector<NodeID> compressed_graph_neighbours;
  for (const NodeID node : graph.nodes()) {
    ASSERT_EQ(graph.degree(node), compressed_graph.degree(node));
    ASSERT_EQ(graph.node_weight(node), compressed_graph.node_weight(node));

    graph.adjacent_nodes(node, [&](const NodeID adjacent_node) {
      graph_neighbours.push_back(adjacent_node);
    });

    compressed_graph.adjacent_nodes(node, [&](const NodeID adjacent_node) {
      compressed_graph_neighbours.push_back(adjacent_node);
    });

    EXPECT_EQ(graph_neighbours.size(), compressed_graph_neighbours.size());

    std::sort(graph_neighbours.begin(), graph_neighbours.end());
    std::sort(compressed_graph_neighbours.begin(), compressed_graph_neighbours.end());

    EXPECT_TRUE(graph_neighbours == compressed_graph_neighbours);

    graph_neighbours.clear();
    compressed_graph_neighbours.clear();
  }
}

template <typename Compressor> void test_graph_compression(Compressor &&compressor) {
  TEST_ON_ALL_GRAPHS2([&](Graph graph) {
    auto &csr_graph = graph.csr_graph();
    const auto compressed_graph = compressor(csr_graph);

    is_correctly_compressed(csr_graph, compressed_graph);
  });
}

TEST(CompressedGraphBuilderTest, csr_graph_compress_test) {
  test_graph_compression([](const CSRGraph &graph) { return compress(graph); });
}

TEST(CompressedGraphBuilderTest, csr_compress_test) {
  test_graph_compression([](CSRGraph &graph) {
    return compress(
        to_span(graph.raw_nodes()),
        to_span(graph.raw_edges()),
        graph.is_node_weighted() ? to_span(graph.raw_node_weights()) : std::span<NodeWeight>(),
        graph.is_edge_weighted() ? to_span(graph.raw_edge_weights()) : std::span<EdgeWeight>()
    );
  });
}

TEST(CompressedGraphBuilderTest, parallel_csr_graph_compress_test) {
  test_graph_compression([](const CSRGraph &graph) { return parallel_compress(graph); });
}

TEST(CompressedGraphBuilderTest, parallel_csr_compress_test) {
  test_graph_compression([](CSRGraph &graph) {
    return parallel_compress(
        to_span(graph.raw_nodes()),
        to_span(graph.raw_edges()),
        graph.is_node_weighted() ? to_span(graph.raw_node_weights()) : std::span<NodeWeight>(),
        graph.is_edge_weighted() ? to_span(graph.raw_edge_weights()) : std::span<EdgeWeight>()
    );
  });
}

TEST(CompressedGraphBuilderTest, compressed_graph_builder_test) {
  test_graph_compression([](const CSRGraph &graph) {
    const bool store_node_weights = graph.is_node_weighted();
    const bool store_edge_weights = graph.is_edge_weighted();

    CompressedGraphBuilder builder(
        graph.n(), graph.m(), store_node_weights, store_edge_weights, graph.sorted()
    );

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    neighbourhood.reserve(graph.max_degree());

    for (const NodeID u : graph.nodes()) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        neighbourhood.emplace_back(v, w);
      });

      builder.add_node(neighbourhood);
      if (store_node_weights) {
        builder.add_node_weight(u, graph.node_weight(u));
      }

      neighbourhood.clear();
    }

    return builder.build();
  });
}

TEST(CompressedGraphBuilderTest, two_pass_parallel_compressed_graph_builder_test) {
  test_graph_compression([](const CSRGraph &graph) {
    const bool store_node_weights = graph.is_node_weighted();
    const bool store_edge_weights = graph.is_edge_weighted();

    ParallelCompressedGraphBuilder builder(
        graph.n(), graph.m(), store_node_weights, store_edge_weights, graph.sorted()
    );

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    neighbourhood.reserve(graph.max_degree());

    for (const NodeID u : graph.nodes()) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        neighbourhood.emplace_back(v, w);
      });

      builder.register_neighborhood(u, neighbourhood);
      neighbourhood.clear();
    }

    builder.compute_offsets();

    for (const NodeID u : graph.nodes()) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        neighbourhood.emplace_back(v, w);
      });

      builder.add_neighborhood(u, neighbourhood);
      if (store_node_weights) {
        builder.add_node_weight(u, graph.node_weight(u));
      }

      neighbourhood.clear();
    }

    return builder.build();
  });
}

TEST(CompressedGraphBuilderTest, single_pass_parallel_compressed_graph_builder_test) {
  test_graph_compression([](const CSRGraph &graph) {
    StaticArray<NodeWeight> node_weights;
    if (graph.is_node_weighted()) {
      node_weights.resize(graph.n(), static_array::noinit);

      tbb::parallel_for<NodeID>(0, graph.n(), [&](const NodeID u) {
        node_weights[u] = graph.node_weight(u);
      });
    }

    const auto fetch_degree = [&](const NodeID u) {
      return graph.degree(u);
    };

    if (graph.is_edge_weighted()) {
      using Edge = std::pair<NodeID, EdgeWeight>;
      const auto fetch_neighborhood = [&](const NodeID u, std::span<Edge> neighborhood) {
        NodeID i = 0;
        graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          neighborhood[i++] = {v, w};
        });
      };

      return parallel_compress_weighted(
          graph.n(),
          graph.m(),
          fetch_degree,
          fetch_neighborhood,
          std::move(node_weights),
          graph.sorted()
      );
    } else {
      const auto fetch_neighborhood = [&](const NodeID u, std::span<NodeID> neighborhood) {
        NodeID i = 0;
        graph.adjacent_nodes(u, [&](const NodeID v) { neighborhood[i++] = v; });
      };

      return parallel_compress(
          graph.n(),
          graph.m(),
          fetch_degree,
          fetch_neighborhood,
          std::move(node_weights),
          graph.sorted()
      );
    }
  });
}

} // namespace kaminpar::shm::testing
