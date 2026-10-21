/*******************************************************************************
 * @file:   shm_metis_io_test.cc
 * @brief:  Unit tests for shared-memory METIS graph IO.
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
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
#include "kaminpar-io/util/file_toker.h"
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
      "2 1\n1\n2\n",
      "2 1\n18446744073709551616\n1\n",
      "2 1 1\n2 18446744073709551616\n1 1\n",
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

TEST(ShmMetisIOTest, readers_reject_total_weight_overflow) {
  const std::string max_node_weight = std::to_string(std::numeric_limits<NodeWeight>::max());
  const std::string max_edge_weight = std::to_string(std::numeric_limits<EdgeWeight>::max());
  const std::array<std::string, 2> malformed_graphs = {
      "2 0 10\n" + max_node_weight + "\n1\n",
      "2 1 1\n2 " + max_edge_weight + "\n1 " + max_edge_weight + "\n",
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

TEST(ShmMetisIOTest, readers_accept_weight_sums_at_type_boundary) {
  TempFile file(".metis");
  const auto max_node_weight = std::numeric_limits<NodeWeight>::max();
  const auto edge_weight = std::numeric_limits<EdgeWeight>::max() / 2;
  std::ostringstream contents;
  contents << "2 1 11\n"
           << max_node_weight - 1 << " 2 " << edge_weight << "\n1 1 " << edge_weight << '\n';
  write_text(file.string(), contents.str());

  for (const bool parallel : {false, true}) {
    for (const bool compress : {false, true}) {
      SCOPED_TRACE(parallel);
      SCOPED_TRACE(compress);
      const auto graph =
          io::metis::read_graph(file.string(), compress, NodeOrdering::NATURAL, parallel);
      ASSERT_TRUE(graph);
      EXPECT_EQ(graph->total_node_weight(), max_node_weight);
      EXPECT_EQ(graph->total_edge_weight(), 2 * edge_weight);
    }
  }
}

TEST(ShmMetisIOTest, readers_check_weight_sums_across_chunks) {
  constexpr NodeID n = 600000;
  const auto max_weight = std::numeric_limits<NodeWeight>::max();
  const auto first_weight = max_weight / 2;
  const auto last_weight = max_weight - first_weight - static_cast<NodeWeight>(n - 2);
  tbb::task_arena arena(4);
  for (const bool overflow : {false, true}) {
    TempFile file(".metis");
    std::ostringstream contents;
    contents << n << " 0 10\n" << first_weight << '\n';
    for (NodeID u = 1; u + 1 < n; ++u) {
      contents << "1\n";
    }
    contents << last_weight + static_cast<NodeWeight>(overflow) << '\n';
    ASSERT_GT(contents.str().size(), 1u << 20);
    write_text(file.string(), contents.str());
    for (const bool parallel : {false, true}) {
      for (const bool compress : {false, true}) {
        SCOPED_TRACE(overflow);
        SCOPED_TRACE(parallel);
        SCOPED_TRACE(compress);
        const auto graph = arena.execute([&] {
          return io::metis::read_graph(file.string(), compress, NodeOrdering::NATURAL, parallel);
        });
        if (overflow) {
          EXPECT_FALSE(graph);
        } else {
          ASSERT_TRUE(graph);
          EXPECT_EQ(graph->n(), n);
          EXPECT_EQ(graph->total_node_weight(), max_weight);
        }
      }
    }
  }
}

TEST(ShmMetisIOTest, parallel_reader_rejects_self_loop_at_global_chunk_offset) {
  constexpr NodeID n = 600000;
  TempFile file(".metis");
  std::ostringstream contents;
  contents << n << " 1\n2\n";
  for (NodeID u = 1; u + 1 < n; ++u) {
    contents << " \n";
  }
  contents << n << '\n';
  ASSERT_GT(contents.str().size(), 1u << 20);
  write_text(file.string(), contents.str());

  tbb::task_arena arena(4);
  for (const bool compress : {false, true}) {
    SCOPED_TRACE(compress);
    const auto graph = arena.execute([&] {
      return io::metis::read_graph(file.string(), compress, NodeOrdering::NATURAL, true);
    });
    EXPECT_FALSE(graph);
  }
}

TEST(MappedFileTokerTest, reads_uint64_boundaries_and_arbitrary_leading_zeros) {
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  const std::array<std::pair<std::string, std::uint64_t>, 8> values = {
      std::pair{"0", 0},
      {"9", 9},
      {"10", 10},
      {"9999999999999999999", 9'999'999'999'999'999'999ull},
      {"18446744073709551614", max - 1},
      {"18446744073709551615", max},
      {std::string(100, '0'), 0},
      {std::string(100, '0') + "18446744073709551615", max},
  };
  for (const auto &[text, expected] : values) {
    SCOPED_TRACE(text);
    TempFile file(".txt");
    write_text(file.string(), text + " 7");
    kaminpar::io::MappedFileToker toker(file.string());
    EXPECT_EQ(toker.scan_uint(), expected);
    EXPECT_EQ(toker.scan_uint(), 7);
    EXPECT_FALSE(toker.valid_position());
  }
}

TEST(MappedFileTokerTest, rejects_invalid_and_overflowing_unsigned_integers) {
  const std::array<std::string, 7> values = {
      "18446744073709551616",
      "99999999999999999999",
      std::string(100, '0') + "18446744073709551616",
      std::string(100, '9'),
      "-1",
      "+1",
      "x",
  };
  for (const std::string &text : values) {
    SCOPED_TRACE(text);
    TempFile file(".txt");
    write_text(file.string(), text);
    kaminpar::io::MappedFileToker toker(file.string());
    EXPECT_THROW(toker.scan_uint(), kaminpar::io::IOException);
  }
}

TEST(MappedFileTokerTest, respects_chunk_bounds_inside_integer) {
  TempFile file(".txt");
  write_text(file.string(), "18446744073709551616");
  kaminpar::io::MappedFileToker parent(file.string());
  kaminpar::io::MappedFileToker toker(parent, 0, 19);
  EXPECT_EQ(toker.scan_uint(), 1'844'674'407'370'955'161ull);
  EXPECT_FALSE(toker.valid_position());
  EXPECT_FALSE(toker.current_is_digit());
  EXPECT_THROW(toker.scan_uint(), kaminpar::io::IOException);
}

TEST(MappedFileTokerTest, preserves_cursor_at_offending_digit_after_overflow) {
  TempFile file(".txt");
  const std::string leading_zeros(100, '0');
  write_text(file.string(), "7 " + leading_zeros + "18446744073709551616");
  kaminpar::io::MappedFileToker toker(file.string());
  ASSERT_EQ(toker.scan_uint(), 7);
  ASSERT_EQ(toker.position(), 2);
  EXPECT_THROW(toker.scan_uint(), kaminpar::io::IOException);
  EXPECT_EQ(toker.position(), 2 + leading_zeros.size() + 19);
  EXPECT_EQ(toker.current(), '6');
}

TEST(MappedFileTokerTest, recognizes_only_ascii_digits) {
  TempFile file(".txt");
  std::string bytes;
  for (int byte = 0; byte <= 255; ++byte) {
    bytes.push_back(static_cast<char>(byte));
  }
  write_text(file.string(), bytes);
  kaminpar::io::MappedFileToker toker(file.string());
  for (int byte = 0; byte <= 255; ++byte) {
    SCOPED_TRACE(byte);
    EXPECT_EQ(toker.current_is_digit(), byte >= '0' && byte <= '9');
    toker.advance();
  }
  EXPECT_FALSE(toker.current_is_digit());
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
