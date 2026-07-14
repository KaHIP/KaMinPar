/*******************************************************************************
 * @file:   shm_metis_io_test.cc
 * @brief:  Unit tests for shared-memory METIS graph IO.
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "kaminpar-io/kaminpar_io.h"
#include "kaminpar-io/metis_parser.h"
#include "tests/io/shm_io_helpers.h"

namespace kaminpar::shm::testing {
namespace {

using namespace io_helpers;

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

TEST(ShmMetisIOTest, readers_read_fixture_files) {
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

TEST(ShmMetisIOTest, dispatch_reads_sequential_and_parallel_formats) {
  const GraphSnapshot expected = rgg16_snapshot(true, true);
  const std::string filename = source_file("tests/io/rgg16-vwgt-adjwgt.metis");

  const auto sequential = io::read_graph(filename, io::GraphFileFormat::METIS);
  ASSERT_TRUE(sequential);
  expect_graph_eq(*sequential, expected);

  const auto parallel = io::read_graph(filename, io::GraphFileFormat::METIS_PARALLEL);
  ASSERT_TRUE(parallel);
  expect_graph_eq(*parallel, expected);
}

TEST(ShmMetisIOTest, parallel_reader_matches_oracle_with_comments_and_weights) {
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

TEST(ShmMetisIOTest, explicit_unit_weights_are_compacted) {
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

TEST(ShmMetisIOTest, readers_skip_declared_node_sizes) {
  TempFile file(".metis");
  write_text(
      file.string(),
      R"(2 1 100
17 2
19 1
)"
  );

  GraphSnapshot expected;
  expected.n = 2;
  expected.m = 2;
  expected.node_weights = {1, 1};
  expected.adjacency = {{{1, 1}}, {{0, 1}}};
  expected.total_node_weight = 2;
  expected.total_edge_weight = 2;

  const auto graph = io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, false);
  ASSERT_TRUE(graph);
  expect_graph_eq(*graph, expected);
}

TEST(ShmMetisIOTest, readers_reject_malformed_graphs) {
  const std::vector<std::string> malformed_graphs = {
      "2 1\n2\n",
      "2 0\n2\n1\n",
      "2 1\n3\n1\n",
      "2 1\n0\n1\n",
      "2 1 2\n2\n1\n",
      "2 1 11\n1 2 0\n1 1 1\n",
  };

  for (const std::string &contents : malformed_graphs) {
    TempFile file(".metis");
    write_text(file.string(), contents);

    for (const bool parallel : {false, true}) {
      for (const bool compress : {false, true}) {
        SCOPED_TRACE(contents);
        SCOPED_TRACE(parallel);
        SCOPED_TRACE(compress);
        EXPECT_FALSE(
            io::metis::read_graph(file.string(), compress, NodeOrdering::NATURAL, parallel)
        );
      }
    }
  }
}

TEST(ShmMetisIOTest, parallel_reader_matches_large_multichunk_oracle) {
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

TEST(ShmMetisIOTest, parallel_reader_cancels_after_multichunk_parse_error) {
  TempFile file(".metis");
  constexpr NodeID n = 2'100'000;
  {
    std::ofstream out(file.string());
    out << n << " 0\n";
    out << "x\n";
    for (NodeID u = 1; u < n; ++u) {
      out << "    \n";
    }
  }

  tbb::task_arena arena(4);
  const auto graph = arena.execute([&] {
    return io::metis::read_graph(file.string(), false, NodeOrdering::NATURAL, true);
  });
  EXPECT_FALSE(graph);
}

} // namespace kaminpar::shm::testing
