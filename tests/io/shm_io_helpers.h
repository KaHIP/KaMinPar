/*******************************************************************************
 * @file:   shm_io_helpers.h
 * @brief:  Shared helpers for shared-memory graph IO tests.
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <unistd.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::testing::io_helpers {

using ::testing::ElementsAreArray;

inline std::string source_file(const char *relative_path) {
  return (std::filesystem::path(KAMINPAR_TEST_SOURCE_DIR) / relative_path).string();
}

struct TempFile {
  explicit TempFile(std::string suffix = "") {
    static std::uint64_t next_id = 0;
    path = std::filesystem::temp_directory_path() /
           ("kaminpar-shm-io-test-" + std::to_string(::getpid()) + "-" + std::to_string(next_id++) +
            suffix);
  }

  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;

  [[nodiscard]] std::string string() const {
    return path.string();
  }

  std::filesystem::path path;
};

struct Edge {
  NodeID target;
  EdgeWeight weight;

  friend bool operator==(const Edge &, const Edge &) = default;
  friend auto operator<=>(const Edge &, const Edge &) = default;
};

struct GraphSnapshot {
  NodeID n = 0;
  EdgeID m = 0;
  bool node_weighted = false;
  bool edge_weighted = false;
  bool sorted = false;
  NodeWeight total_node_weight = 0;
  EdgeWeight total_edge_weight = 0;
  std::vector<NodeWeight> node_weights;
  std::vector<std::vector<Edge>> adjacency;
};

inline GraphSnapshot make_snapshot(const Graph &graph) {
  GraphSnapshot snapshot;
  reified(graph, [&](const auto &concrete_graph) {
    snapshot.n = concrete_graph.n();
    snapshot.m = concrete_graph.m();
    snapshot.node_weighted = concrete_graph.is_node_weighted();
    snapshot.edge_weighted = concrete_graph.is_edge_weighted();
    snapshot.sorted = concrete_graph.sorted();
    snapshot.total_node_weight = concrete_graph.total_node_weight();
    snapshot.total_edge_weight = concrete_graph.total_edge_weight();

    snapshot.node_weights.resize(snapshot.n);
    snapshot.adjacency.resize(snapshot.n);
    for (const NodeID u : concrete_graph.nodes()) {
      snapshot.node_weights[u] = concrete_graph.node_weight(u);
      concrete_graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        snapshot.adjacency[u].push_back({.target = v, .weight = weight});
      });
      std::sort(snapshot.adjacency[u].begin(), snapshot.adjacency[u].end());
    }
  });

  return snapshot;
}

inline void expect_snapshot_eq(const GraphSnapshot &actual, const GraphSnapshot &expected) {
  EXPECT_EQ(actual.n, expected.n);
  EXPECT_EQ(actual.m, expected.m);
  EXPECT_EQ(actual.node_weighted, expected.node_weighted);
  EXPECT_EQ(actual.edge_weighted, expected.edge_weighted);
  EXPECT_EQ(actual.total_node_weight, expected.total_node_weight);
  EXPECT_EQ(actual.total_edge_weight, expected.total_edge_weight);
  EXPECT_THAT(actual.node_weights, ElementsAreArray(expected.node_weights));
  ASSERT_EQ(actual.adjacency.size(), expected.adjacency.size());
  for (NodeID u = 0; u < actual.adjacency.size(); ++u) {
    EXPECT_THAT(actual.adjacency[u], ElementsAreArray(expected.adjacency[u])) << "node " << u;
  }
}

inline void expect_graph_eq(const Graph &actual, const GraphSnapshot &expected) {
  expect_snapshot_eq(make_snapshot(actual), expected);
}

inline std::vector<std::vector<Edge>> canonicalize_by_node_weight(const GraphSnapshot &snapshot) {
  // This is intentionally only used with weighted_test_snapshot(), whose node weights are dense
  // identity labels in [0, n). This keeps the comparison compact for degree-bucket reorderings.
  std::vector<std::vector<Edge>> adjacency_by_weight(snapshot.n);
  for (NodeID u = 0; u < snapshot.n; ++u) {
    const NodeID u_key = static_cast<NodeID>(snapshot.node_weights[u]);
    for (const Edge edge : snapshot.adjacency[u]) {
      adjacency_by_weight[u_key].push_back(
          {.target = static_cast<NodeID>(snapshot.node_weights[edge.target]), .weight = edge.weight}
      );
    }
    std::sort(adjacency_by_weight[u_key].begin(), adjacency_by_weight[u_key].end());
  }
  return adjacency_by_weight;
}

inline void
expect_graph_eq_by_unique_node_weight(const Graph &actual, const GraphSnapshot &expected) {
  const GraphSnapshot actual_snapshot = make_snapshot(actual);
  EXPECT_EQ(actual_snapshot.n, expected.n);
  EXPECT_EQ(actual_snapshot.m, expected.m);
  EXPECT_TRUE(actual_snapshot.node_weighted);
  EXPECT_EQ(actual_snapshot.edge_weighted, expected.edge_weighted);
  EXPECT_EQ(actual_snapshot.total_node_weight, expected.total_node_weight);
  EXPECT_EQ(actual_snapshot.total_edge_weight, expected.total_edge_weight);

  std::vector<NodeWeight> actual_weights = actual_snapshot.node_weights;
  std::vector<NodeWeight> expected_weights = expected.node_weights;
  std::sort(actual_weights.begin(), actual_weights.end());
  std::sort(expected_weights.begin(), expected_weights.end());
  EXPECT_THAT(actual_weights, ElementsAreArray(expected_weights));
  EXPECT_THAT(
      canonicalize_by_node_weight(actual_snapshot),
      ElementsAreArray(canonicalize_by_node_weight(expected))
  );
}

inline GraphSnapshot rgg16_snapshot(const bool has_node_weights, const bool has_edge_weights) {
  static const std::array<std::vector<Edge>, 16> kWeightedTopology = {
      std::vector<Edge>{{1, 1}, {8, 21}, {15, 16}},
      std::vector<Edge>{{0, 1}, {2, 2}, {9, 22}},
      std::vector<Edge>{{1, 2}, {3, 3}, {10, 23}},
      std::vector<Edge>{{2, 3}, {4, 4}, {11, 24}},
      std::vector<Edge>{{3, 4}, {5, 5}, {12, 25}},
      std::vector<Edge>{{4, 5}, {6, 6}, {13, 26}},
      std::vector<Edge>{{5, 6}, {7, 7}, {14, 27}},
      std::vector<Edge>{{6, 7}, {8, 8}, {15, 28}},
      std::vector<Edge>{{0, 21}, {7, 8}, {9, 9}},
      std::vector<Edge>{{1, 22}, {8, 9}, {10, 10}},
      std::vector<Edge>{{2, 23}, {9, 10}, {11, 11}},
      std::vector<Edge>{{3, 24}, {10, 11}, {12, 12}},
      std::vector<Edge>{{4, 25}, {11, 12}, {13, 13}},
      std::vector<Edge>{{5, 26}, {12, 13}, {14, 14}},
      std::vector<Edge>{{6, 27}, {13, 14}, {15, 15}},
      std::vector<Edge>{{0, 16}, {7, 28}, {14, 15}},
  };

  GraphSnapshot snapshot;
  snapshot.n = 16;
  snapshot.m = 48;
  snapshot.node_weighted = has_node_weights;
  snapshot.edge_weighted = has_edge_weights;
  snapshot.node_weights.resize(snapshot.n);
  snapshot.adjacency.assign(kWeightedTopology.begin(), kWeightedTopology.end());

  for (NodeID u = 0; u < snapshot.n; ++u) {
    snapshot.node_weights[u] = has_node_weights ? static_cast<NodeWeight>(u + 1) : 1;
    if (!has_edge_weights) {
      for (Edge &edge : snapshot.adjacency[u]) {
        edge.weight = 1;
      }
    }
    std::sort(snapshot.adjacency[u].begin(), snapshot.adjacency[u].end());
  }

  snapshot.total_node_weight = std::accumulate(
      snapshot.node_weights.begin(), snapshot.node_weights.end(), static_cast<NodeWeight>(0)
  );
  snapshot.total_edge_weight = 0;
  for (const auto &adjacency : snapshot.adjacency) {
    for (const Edge edge : adjacency) {
      snapshot.total_edge_weight += edge.weight;
    }
  }

  return snapshot;
}

inline GraphSnapshot weighted_test_snapshot() {
  GraphSnapshot snapshot;
  snapshot.n = 7;
  snapshot.node_weighted = true;
  snapshot.edge_weighted = true;
  snapshot.node_weights = {0, 1, 2, 3, 4, 5, 6};
  snapshot.adjacency = {
      {{1, 7}, {2, 11}, {3, 13}, {4, 17}, {5, 19}, {6, 23}},
      {{0, 7}, {2, 29}},
      {{0, 11}, {1, 29}, {3, 31}, {4, 37}},
      {{0, 13}, {2, 31}},
      {{0, 17}, {2, 37}, {5, 41}},
      {{0, 19}, {4, 41}, {6, 43}},
      {{0, 23}, {5, 43}},
  };
  for (auto &adjacency : snapshot.adjacency) {
    std::sort(adjacency.begin(), adjacency.end());
    snapshot.m += static_cast<EdgeID>(adjacency.size());
    for (const Edge edge : adjacency) {
      snapshot.total_edge_weight += edge.weight;
    }
  }
  snapshot.total_node_weight = std::accumulate(
      snapshot.node_weights.begin(), snapshot.node_weights.end(), static_cast<NodeWeight>(0)
  );
  return snapshot;
}

inline Graph make_graph(const GraphSnapshot &snapshot) {
  StaticArray<EdgeID> nodes(snapshot.n + 1, static_array::noinit);
  StaticArray<NodeID> edges(snapshot.m, static_array::noinit);
  StaticArray<EdgeWeight> edge_weights;
  if (snapshot.edge_weighted) {
    edge_weights.resize(snapshot.m, static_array::noinit);
  }
  StaticArray<NodeWeight> node_weights;
  if (snapshot.node_weighted) {
    node_weights.resize(snapshot.n, static_array::noinit);
  }

  EdgeID cur_edge = 0;
  for (NodeID u = 0; u < snapshot.n; ++u) {
    nodes[u] = cur_edge;
    if (snapshot.node_weighted) {
      node_weights[u] = snapshot.node_weights[u];
    }
    for (const Edge edge : snapshot.adjacency[u]) {
      edges[cur_edge] = edge.target;
      if (snapshot.edge_weighted) {
        edge_weights[cur_edge] = edge.weight;
      }
      ++cur_edge;
    }
  }
  nodes[snapshot.n] = cur_edge;

  return Graph(
      std::make_unique<CSRGraph>(
          std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
      )
  );
}

template <typename T> void write_binary(std::ofstream &out, const T value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

inline void write_text(const std::string &filename, const std::string &contents) {
  std::ofstream out(filename);
  out << contents;
}

inline std::vector<NodeID> flattened_edges(const GraphSnapshot &snapshot) {
  std::vector<NodeID> edges;
  edges.reserve(snapshot.m);
  for (const auto &adjacency : snapshot.adjacency) {
    for (const Edge edge : adjacency) {
      edges.push_back(edge.target);
    }
  }
  return edges;
}

inline std::string read_file(const std::string &filename) {
  std::ifstream in(filename);
  std::ostringstream contents;
  contents << in.rdbuf();
  return contents.str();
}

} // namespace kaminpar::shm::testing::io_helpers
