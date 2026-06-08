/*******************************************************************************
 * @file:   shm_io_test.cc
 * @brief:  Unit tests for shared-memory graph IO.
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "kaminpar-io/graph_compression_binary.h"
#include "kaminpar-io/kaminpar_io.h"
#include "kaminpar-io/metis_parser.h"
#include "kaminpar-io/parhip_parser.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::testing {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Eq;

std::string source_file(const char *relative_path) {
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

GraphSnapshot make_snapshot(const Graph &graph) {
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

void expect_snapshot_eq(const GraphSnapshot &actual, const GraphSnapshot &expected) {
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

void expect_graph_eq(const Graph &actual, const GraphSnapshot &expected) {
  expect_snapshot_eq(make_snapshot(actual), expected);
}

std::vector<std::vector<Edge>> canonicalize_by_node_weight(const GraphSnapshot &snapshot) {
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

void expect_graph_eq_by_unique_node_weight(const Graph &actual, const GraphSnapshot &expected) {
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

GraphSnapshot rgg16_snapshot(const bool has_node_weights, const bool has_edge_weights) {
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

GraphSnapshot weighted_test_snapshot() {
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

Graph make_graph(const GraphSnapshot &snapshot) {
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

void write_text(const std::string &filename, const std::string &contents) {
  std::ofstream out(filename);
  out << contents;
}

std::vector<NodeID> flattened_edges(const GraphSnapshot &snapshot) {
  std::vector<NodeID> edges;
  edges.reserve(snapshot.m);
  for (const auto &adjacency : snapshot.adjacency) {
    for (const Edge edge : adjacency) {
      edges.push_back(edge.target);
    }
  }
  return edges;
}

std::string generated_metis_fixture() {
  return R"(% leading comment
9 9 11
% line before first vertex
2 2 5 3 7
3 1 5 3 11 4 13
5 1 7 2 11
7 2 13 5 17
11 4 17 6 19 7 23
13 5 19 8 29
17 5 23 8 31
19 6 29 7 31
23
% trailing comment
)";
}

GraphSnapshot generated_metis_snapshot() {
  GraphSnapshot snapshot;
  snapshot.n = 9;
  snapshot.m = 18;
  snapshot.node_weighted = true;
  snapshot.edge_weighted = true;
  snapshot.node_weights = {2, 3, 5, 7, 11, 13, 17, 19, 23};
  snapshot.adjacency = {
      {{1, 5}, {2, 7}},
      {{0, 5}, {2, 11}, {3, 13}},
      {{0, 7}, {1, 11}},
      {{1, 13}, {4, 17}},
      {{3, 17}, {5, 19}, {6, 23}},
      {{4, 19}, {7, 29}},
      {{4, 23}, {7, 31}},
      {{5, 29}, {6, 31}},
      {},
  };
  for (auto &adjacency : snapshot.adjacency) {
    std::sort(adjacency.begin(), adjacency.end());
    for (const Edge edge : adjacency) {
      snapshot.total_edge_weight += edge.weight;
    }
  }
  snapshot.total_node_weight = std::accumulate(
      snapshot.node_weights.begin(), snapshot.node_weights.end(), static_cast<NodeWeight>(0)
  );
  return snapshot;
}

std::string explicit_unit_weight_metis_fixture() {
  return R"(4 2 11
1 2 1
1 1 1
1 4 1
1 3 1
)";
}

GraphSnapshot explicit_unit_weight_snapshot() {
  GraphSnapshot snapshot;
  snapshot.n = 4;
  snapshot.m = 4;
  snapshot.node_weighted = false;
  snapshot.edge_weighted = false;
  snapshot.node_weights = {1, 1, 1, 1};
  snapshot.adjacency = {
      {{1, 1}},
      {{0, 1}},
      {{3, 1}},
      {{2, 1}},
  };
  snapshot.total_node_weight = 4;
  snapshot.total_edge_weight = 4;
  return snapshot;
}

struct LargeMetisFixture {
  std::string contents;
  GraphSnapshot snapshot;
};

LargeMetisFixture make_large_metis_fixture() {
  constexpr NodeID n = 70000;
  GraphSnapshot snapshot;
  snapshot.n = n;
  snapshot.node_weighted = true;
  snapshot.edge_weighted = true;
  snapshot.node_weights.resize(n);
  snapshot.adjacency.resize(n);

  std::vector<std::tuple<NodeID, NodeID, EdgeWeight>> undirected_edges;
  undirected_edges.reserve(2 * n);
  for (NodeID u = 0; u + 1 < n; ++u) {
    undirected_edges.emplace_back(u, u + 1, static_cast<EdgeWeight>(2 + (u % 31)));
  }
  for (NodeID u = 0; u + 17 < n; u += 11) {
    undirected_edges.emplace_back(u, u + 17, static_cast<EdgeWeight>(37 + (u % 17)));
  }

  for (const auto [u, v, weight] : undirected_edges) {
    snapshot.adjacency[u].push_back({.target = v, .weight = weight});
    snapshot.adjacency[v].push_back({.target = u, .weight = weight});
    snapshot.total_edge_weight += 2 * weight;
  }
  snapshot.m = static_cast<EdgeID>(2 * undirected_edges.size());

  std::ostringstream out;
  out << "% generated large fixture\n" << n << ' ' << undirected_edges.size() << " 11\n";
  for (NodeID u = 0; u < n; ++u) {
    snapshot.node_weights[u] = static_cast<NodeWeight>(1 + (u % 127));
    snapshot.total_node_weight += snapshot.node_weights[u];

    if (u % 10000 == 0) {
      out << "% comment before vertex " << u << '\n';
    }
    out << snapshot.node_weights[u];
    std::sort(snapshot.adjacency[u].begin(), snapshot.adjacency[u].end());
    for (const Edge edge : snapshot.adjacency[u]) {
      out << ' ' << (edge.target + 1) << ' ' << edge.weight;
    }
    out << '\n';
  }
  out << "% trailing comment\n";

  return {.contents = std::move(out).str(), .snapshot = std::move(snapshot)};
}

std::string read_file(const std::string &filename) {
  std::ifstream in(filename);
  std::ostringstream contents;
  contents << in.rdbuf();
  return contents.str();
}

constexpr const char *kMetisFiles[] = {
    "tests/io/rgg16.metis",
    "tests/io/rgg16-vwgt.metis",
    "tests/io/rgg16-adjwgt.metis",
    "tests/io/rgg16-vwgt-adjwgt.metis",
};

constexpr std::array<std::pair<bool, bool>, 4> kMetisWeightFlags = {
    std::pair{false, false},
    std::pair{true, false},
    std::pair{false, true},
    std::pair{true, true},
};

} // namespace

TEST(ShmIOTest, metis_readers_read_fixture_files) {
  for (std::size_t i = 0; i < kMetisWeightFlags.size(); ++i) {
    const auto [has_node_weights, has_edge_weights] = kMetisWeightFlags[i];
    const GraphSnapshot expected = rgg16_snapshot(has_node_weights, has_edge_weights);
    const std::string filename = source_file(kMetisFiles[i]);
    SCOPED_TRACE(filename);

    const auto csr = io::metis::read_graph(filename, false, NodeOrdering::NATURAL, false);
    ASSERT_TRUE(csr);
    EXPECT_TRUE(csr->is_csr());
    expect_graph_eq(*csr, expected);

    const auto compressed = io::metis::read_graph(filename, true, NodeOrdering::NATURAL, false);
    ASSERT_TRUE(compressed);
    EXPECT_TRUE(compressed->is_compressed());
    expect_graph_eq(*compressed, expected);

    const auto parallel = io::metis::read_graph(filename, false, NodeOrdering::NATURAL, true);
    ASSERT_TRUE(parallel);
    EXPECT_TRUE(parallel->is_csr());
    expect_graph_eq(*parallel, expected);

    const auto parallel_compressed =
        io::metis::read_graph(filename, true, NodeOrdering::NATURAL, true);
    ASSERT_TRUE(parallel_compressed);
    EXPECT_TRUE(parallel_compressed->is_compressed());
    expect_graph_eq(*parallel_compressed, expected);
  }
}

TEST(ShmIOTest, metis_dispatch_reads_sequential_and_parallel_formats) {
  const GraphSnapshot expected = rgg16_snapshot(true, true);
  const std::string filename = source_file("tests/io/rgg16-vwgt-adjwgt.metis");

  const auto sequential = io::read_graph(filename, io::GraphFileFormat::METIS);
  ASSERT_TRUE(sequential);
  expect_graph_eq(*sequential, expected);

  const auto parallel = io::read_graph(filename, io::GraphFileFormat::METIS_PARALLEL);
  ASSERT_TRUE(parallel);
  expect_graph_eq(*parallel, expected);
}

TEST(ShmIOTest, metis_parallel_reader_matches_oracle_with_comments_and_weights) {
  TempFile file(".metis");
  write_text(file.string(), generated_metis_fixture());
  const GraphSnapshot expected = generated_metis_snapshot();

  const auto sequential = io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, false);
  ASSERT_TRUE(sequential);
  expect_graph_eq(*sequential, expected);

  for (const int threads : {1, 2, 4}) {
    SCOPED_TRACE(threads);
    tbb::task_arena arena(threads);
    const auto parallel = arena.execute([&] {
      return io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, true);
    });
    ASSERT_TRUE(parallel);
    expect_snapshot_eq(make_snapshot(*parallel), make_snapshot(*sequential));
  }
}

TEST(ShmIOTest, metis_explicit_unit_weights_are_compacted) {
  TempFile file(".metis");
  write_text(file.string(), explicit_unit_weight_metis_fixture());
  const GraphSnapshot expected = explicit_unit_weight_snapshot();

  for (const bool parallel : {false, true}) {
    SCOPED_TRACE(parallel);
    const auto graph = io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, parallel);
    ASSERT_TRUE(graph);
    EXPECT_FALSE(graph->is_node_weighted());
    EXPECT_FALSE(graph->is_edge_weighted());
    expect_graph_eq(*graph, expected);
  }
}

TEST(ShmIOTest, metis_parallel_reader_matches_large_multichunk_oracle) {
  TempFile file(".metis");
  LargeMetisFixture fixture = make_large_metis_fixture();
  ASSERT_GT(fixture.contents.size(), 1u << 20);
  write_text(file.string(), fixture.contents);

  const auto sequential = io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, false);
  ASSERT_TRUE(sequential);
  expect_graph_eq(*sequential, fixture.snapshot);

  const std::array<int, 4> thread_counts = {
      1, 2, 4, std::max(1, tbb::this_task_arena::max_concurrency())
  };
  for (const int threads : thread_counts) {
    SCOPED_TRACE(threads);
    tbb::task_arena arena(threads);
    const auto parallel = arena.execute([&] {
      return io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, true);
    });
    ASSERT_TRUE(parallel);
    expect_graph_eq(*parallel, fixture.snapshot);
  }
}

TEST(ShmIOTest, parhip_direct_fixtures_read_as_csr_and_compressed) {
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

TEST(ShmIOTest, parhip_external_degree_bucket_read_preserves_weight_identified_graph) {
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

TEST(ShmIOTest, parhip_write_graph_roundtrips_and_uses_expected_header) {
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

TEST(ShmIOTest, compressed_binary_roundtrips_compressed_graphs) {
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

TEST(ShmIOTest, compressed_binary_rejects_missing_file_and_wrong_magic) {
  TempFile missing(".compressed");
  EXPECT_FALSE(io::read_graph(missing.string(), io::GraphFileFormat::COMPRESSED));

  TempFile wrong_magic(".compressed");
  {
    std::ofstream out(wrong_magic.string(), std::ios::binary);
    write_binary<std::uint64_t>(out, 0x123456789abcdef0ull);
  }
  EXPECT_FALSE(io::read_graph(wrong_magic.string(), io::GraphFileFormat::COMPRESSED));
}

TEST(ShmIOTest, partition_and_block_size_helpers_roundtrip) {
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

TEST(ShmIOTest, write_remapping_writes_one_node_per_line) {
  TempFile file(".map");
  const std::vector<NodeID> remapping = {4, 2, 0, 3, 1};
  io::write_remapping(file.string(), remapping);
  EXPECT_THAT(read_file(file.string()), Eq("4\n2\n0\n3\n1\n"));
}

} // namespace kaminpar::shm::testing
