/*******************************************************************************
 * @file:   distributed_compressed_graph_test.cc
 * @author: Daniel Salwasser
 * @date:   08.06.2024
 * @brief:  Unit tests for the distributed compressed graph.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dist/distributed_graph_factories.h"

#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

#define TEST_ON_ALL_GRAPHS(test_function)                                                          \
  test_function(testing::make_csr_empty_graph());                                                  \
  test_function(testing::make_csr_circle_graph());                                                 \
  test_function(testing::make_csr_path(1000));                                                     \
  test_function(testing::make_csr_isolated_nodes_graph(1000));                                     \
  test_function(testing::make_csr_isolated_edges_graph(1000));                                     \
  test_function(testing::make_csr_cut_edge_graph(1000));                                           \
  test_function(testing::make_csr_circle_clique_graph(1000));                                      \
  test_function(testing::make_csr_local_complete_graph(100));                                      \
  test_function(testing::make_csr_local_complete_bipartite_graph(100));                            \
  test_function(testing::make_csr_global_complete_graph(100));

namespace kaminpar::dist {

template <typename T>
[[nodiscard]] static bool operator==(const IotaRange<T> &a, const IotaRange<T> &b) {
  return a.begin() == b.begin() && a.end() == b.end();
}

[[nodiscard]] DistributedCompressedGraph compress(const DistributedCSRGraph &graph) {
  const mpi::PEID rank = mpi::get_comm_rank(graph.communicator());

  StaticArray<GlobalNodeID> node_distribution(
      graph.node_distribution().begin(), graph.node_distribution().end()
  );
  StaticArray<GlobalEdgeID> edge_distribution(
      graph.edge_distribution().begin(), graph.edge_distribution().end()
  );

  CompactGhostNodeMappingBuilder mapper(rank, node_distribution);
  CompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight> builder(
      graph.n(), graph.m(), graph.is_edge_weighted()
  );

  const NodeID first_node = node_distribution[rank];

  const auto &raw_node_weights = graph.raw_nodes();

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID adjacent_node) {
      const EdgeWeight edge_weight = graph.is_edge_weighted() ? graph.edge_weight(e) : 1;

      if (graph.is_owned_node(adjacent_node)) {
        neighbourhood.emplace_back(adjacent_node, edge_weight);
      } else {
        const GlobalNodeID original_adjacent_node = graph.local_to_global_node(adjacent_node);
        neighbourhood.emplace_back(mapper.new_ghost_node(original_adjacent_node), edge_weight);
      }
    });

    builder.add(u, neighbourhood);
    neighbourhood.clear();
  }

  StaticArray<NodeWeight> node_weights;
  if (graph.is_node_weighted()) {
    node_weights.resize(graph.n() + mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        node_weights[u] = raw_node_weights[first_node + u];
      }
    });
  }

  DistributedCompressedGraph compressed_graph(
      std::move(node_distribution),
      std::move(edge_distribution),
      builder.build(),
      std::move(node_weights),
      mapper.finalize(),
      graph.sorted(),
      graph.communicator()
  );

  // Fill in ghost node weights
  if (graph.is_node_weighted()) {
    graph::synchronize_ghost_node_weights(compressed_graph);
  }

  return compressed_graph;
}

static void test_compressed_graph_size(const DistributedCSRGraph &graph) {
  const mpi::PEID size = mpi::get_comm_size(graph.communicator());

  const auto compressed_graph = compress(graph);

  EXPECT_EQ(graph.global_n(), compressed_graph.global_n());
  EXPECT_EQ(graph.global_m(), compressed_graph.global_m());

  EXPECT_EQ(graph.n(), compressed_graph.n());
  EXPECT_EQ(graph.m(), compressed_graph.m());

  EXPECT_EQ(graph.ghost_n(), compressed_graph.ghost_n());
  EXPECT_EQ(graph.total_n(), compressed_graph.total_n());

  EXPECT_EQ(graph.offset_n(), compressed_graph.offset_n());
  EXPECT_EQ(graph.offset_m(), compressed_graph.offset_m());

  for (mpi::PEID pe = 0; pe < size; ++pe) {
    EXPECT_EQ(graph.n(pe), compressed_graph.n(pe));
    EXPECT_EQ(graph.m(pe), compressed_graph.m(pe));

    EXPECT_EQ(graph.offset_n(pe), compressed_graph.offset_n(pe));
    EXPECT_EQ(graph.offset_m(pe), compressed_graph.offset_m(pe));
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_size) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_size);
}

static void test_compressed_graph_node_ownership(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  for (const NodeID u : IotaRange<GlobalNodeID>(0, graph.global_n())) {
    EXPECT_EQ(graph.is_owned_global_node(u), compressed_graph.is_owned_global_node(u));
    EXPECT_EQ(graph.contains_global_node(u), compressed_graph.contains_global_node(u));
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_node_ownership) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_node_ownership);
}

static void test_compressed_graph_node_type(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  for (const NodeID u : graph.all_nodes()) {
    EXPECT_EQ(graph.is_ghost_node(u), compressed_graph.is_ghost_node(u));
    EXPECT_EQ(graph.is_owned_node(u), compressed_graph.is_owned_node(u));
    EXPECT_EQ(graph.local_to_global_node(u), compressed_graph.local_to_global_node(u));
  }

  for (const NodeID u : graph.ghost_nodes()) {
    EXPECT_EQ(graph.ghost_owner(u), compressed_graph.ghost_owner(u));
  }

  for (const NodeID u : IotaRange<GlobalNodeID>(0, graph.global_n())) {
    if (graph.contains_global_node(u)) {
      EXPECT_EQ(graph.global_to_local_node(u), compressed_graph.global_to_local_node(u));
    }
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_node_type) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_node_type);
}

static void test_compressed_graph_iterators(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  EXPECT_TRUE(graph.nodes() == compressed_graph.nodes());
  EXPECT_TRUE(graph.ghost_nodes() == compressed_graph.ghost_nodes());
  EXPECT_TRUE(graph.all_nodes() == compressed_graph.all_nodes());

  EXPECT_TRUE(graph.edges() == compressed_graph.edges());
  for (const NodeID u : graph.nodes()) {
    EXPECT_TRUE(graph.incident_edges(u) == compressed_graph.incident_edges(u));
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_iterators) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_iterators);
}

static void test_compressed_graph_cached_inter_pe_metrics(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  const mpi::PEID size = mpi::get_comm_size(graph.communicator());
  for (mpi::PEID pe = 0; pe < size; ++pe) {
    EXPECT_EQ(graph.edge_cut_to_pe(pe), compressed_graph.edge_cut_to_pe(pe));
    EXPECT_EQ(graph.comm_vol_to_pe(pe), compressed_graph.comm_vol_to_pe(pe));
  }

  EXPECT_EQ(graph.communicator(), compressed_graph.communicator());
}

TEST(DistributedCompressedGraphTest, compressed_graph_cached_inter_pe_metrics) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_cached_inter_pe_metrics);
}

static void test_compressed_graph_degree_operation(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  for (const NodeID u : graph.nodes()) {
    EXPECT_EQ(graph.degree(u), compressed_graph.degree(u));
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_degree_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_degree_operation);
}

static void test_compressed_graph_adjacent_nodes_operation(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  std::vector<NodeID> graph_neighbours;
  std::vector<NodeID> compressed_graph_neighbours;
  for (const NodeID u : graph.nodes()) {
    graph.adjacent_nodes(u, [&](const NodeID v) { graph_neighbours.push_back(v); });

    compressed_graph.adjacent_nodes(u, [&](const NodeID v) {
      compressed_graph_neighbours.push_back(v);
    });

    EXPECT_EQ(graph_neighbours.size(), compressed_graph_neighbours.size());

    std::sort(graph_neighbours.begin(), graph_neighbours.end());
    std::sort(compressed_graph_neighbours.begin(), compressed_graph_neighbours.end());
    EXPECT_TRUE(graph_neighbours == compressed_graph_neighbours);

    graph_neighbours.clear();
    compressed_graph_neighbours.clear();
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_adjacent_nodes_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_adjacent_nodes_operation);
}

static void test_compressed_graph_neighbors_operation(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  std::vector<EdgeID> graph_incident_edges;
  std::vector<NodeID> graph_adjacent_node;
  std::vector<EdgeID> compressed_graph_incident_edges;
  std::vector<NodeID> compressed_graph_adjacent_node;
  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      graph_incident_edges.push_back(e);
      graph_adjacent_node.push_back(v);
    });

    compressed_graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      compressed_graph_incident_edges.push_back(e);
      compressed_graph_adjacent_node.push_back(v);
    });

    EXPECT_EQ(graph_incident_edges.size(), compressed_graph_incident_edges.size());

    std::sort(graph_incident_edges.begin(), graph_incident_edges.end());
    std::sort(graph_adjacent_node.begin(), graph_adjacent_node.end());
    std::sort(compressed_graph_incident_edges.begin(), compressed_graph_incident_edges.end());
    std::sort(compressed_graph_adjacent_node.begin(), compressed_graph_adjacent_node.end());
    EXPECT_TRUE(graph_incident_edges == compressed_graph_incident_edges);
    EXPECT_TRUE(graph_adjacent_node == compressed_graph_adjacent_node);

    graph_incident_edges.clear();
    graph_adjacent_node.clear();
    compressed_graph_incident_edges.clear();
    compressed_graph_adjacent_node.clear();
  }
}

TEST(DistributedCompressedGraphTest, compressed_graph_neighbors_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_neighbors_operation);
}

static void test_compressed_graph_neighbors_limit_operation(const DistributedCSRGraph &graph) {
  const auto compressed_graph = compress(graph);

  for (const NodeID u : graph.nodes()) {
    const NodeID max_neighbor_count = std::max<NodeID>(1, graph.degree(u) / 2);

    NodeID graph_num_neighbors_visited = 0;
    graph.neighbors(u, max_neighbor_count, [&](EdgeID, NodeID) {
      graph_num_neighbors_visited += 1;
    });

    NodeID compressed_graph_num_neighbors_visited = 0;
    compressed_graph.neighbors(u, max_neighbor_count, [&](EdgeID, NodeID) {
      compressed_graph_num_neighbors_visited += 1;
    });

    EXPECT_EQ(graph_num_neighbors_visited, compressed_graph_num_neighbors_visited);
  }
}

TEST(CompressedGraphTest, compressed_graph_neighbors_limit_operation) {
  TEST_ON_ALL_GRAPHS(test_compressed_graph_neighbors_limit_operation);
}

} // namespace kaminpar::dist
