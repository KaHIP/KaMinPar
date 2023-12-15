/*******************************************************************************
 * Graph compression benchmark for the shared-memory algorithm.
 *
 * @file:   shm_compressed_graph_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   12.11.2023
 ******************************************************************************/
#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

static std::string to_megabytes(std::size_t bytes) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << (bytes / (float)(1024 * 1024));
  return stream.str();
}

template <typename T> static bool operator!=(const IotaRange<T> &a, const IotaRange<T> &b) {
  return a.begin() != b.begin() || a.end() != b.end();
};

// See https://github.com/google/benchmark/blob/main/include/benchmark/benchmark.h
template <class T> static inline void do_not_optimize(T value) {
  asm volatile("" : "+m"(value) : : "memory");
}

template <typename Graph> static void benchmark_degree(const Graph &graph) {
  SCOPED_HEAP_PROFILER("Degree");
  SCOPED_TIMER("Degree");

  for (const auto node : graph.nodes()) {
    do_not_optimize(graph.degree(node));
  }
}

template <typename Graph> static void benchmark_incident_edges(const Graph &graph) {
  SCOPED_HEAP_PROFILER("Incident Edges");
  SCOPED_TIMER("Incident Edges");

  for (const auto node : graph.nodes()) {
    for (const auto incident_edge : graph.incident_edges(node)) {
      do_not_optimize(incident_edge);
    }
  }
}

template <typename Graph> static void benchmark_adjacent_nodes(const Graph &graph) {
  SCOPED_HEAP_PROFILER("Adjacent Nodes");
  SCOPED_TIMER("Adjacent Nodes");

  for (const auto node : graph.nodes()) {
    graph.adjacent_nodes(node, [&](const auto adjacent_node) { do_not_optimize(adjacent_node); });
  }
}

template <typename Graph> static void benchmark_neighbors(const Graph &graph) {
  SCOPED_HEAP_PROFILER("Neighbors");
  SCOPED_TIMER("Neighbors");

  for (const auto node : graph.nodes()) {
    graph.neighbors(node, [](const auto incident_edge, const auto adjacent_node) {
      do_not_optimize(incident_edge);
      do_not_optimize(adjacent_node);
    });
  }
}

template <typename Graph> static void benchmark_pfor_neighbors(const Graph &graph) {
  SCOPED_HEAP_PROFILER("Parallel For Neighbors");
  SCOPED_TIMER("Parallel For Neighbors");

  for (const auto node : graph.nodes()) {
    graph.pfor_neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        [](const auto incident_edge, const auto adjacent_node) {
          do_not_optimize(incident_edge);
          do_not_optimize(adjacent_node);
        }
    );
  }
}

static void expect_equal_size(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  if (graph.n() != compressed_graph.n()) {
    LOG_ERROR << "The uncompressed graph has " << graph.n()
              << " nodes and the compressed graph has " << compressed_graph.n() << " nodes!";
    return;
  }

  if (graph.m() != compressed_graph.m()) {
    LOG_ERROR << "The uncompressed graph has " << graph.m()
              << " edges and the compressed graph has " << compressed_graph.m() << " edges!";
    return;
  }
}

static void expect_equal_nodes(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  if (graph.nodes() != compressed_graph.nodes()) {
    LOG_ERROR << "The nodes of the compressed and uncompressed graph do not match!";
    return;
  }
}

static void expect_equal_edges(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  if (graph.edges() != compressed_graph.edges()) {
    LOG_ERROR << "The edges of the compressed and uncompressed graph do not match!";
    return;
  }
}

static void expect_equal_degree(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  for (const auto node : graph.nodes()) {
    if (graph.degree(node) != compressed_graph.degree(node)) {
      LOG_ERROR << "The node " << node << " has degree " << compressed_graph.degree(node)
                << " in the compressed graph and degree" << graph.degree(node)
                << " in the uncompressed graph!";
      return;
    }
  }
}

static void
expect_equal_incident_edges(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  for (const auto node : graph.nodes()) {
    if (graph.incident_edges(node) != compressed_graph.incident_edges(node)) {
      LOG_ERROR << "The incident edges of node " << node
                << " in the compressed and uncompressed graph do not match!";
      return;
    }
  }
}

static void
expect_equal_adjacent_nodes(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  std::vector<NodeID> graph_neighbours;
  std::vector<NodeID> compressed_graph_neighbours;

  for (const NodeID node : graph.nodes()) {
    graph.adjacent_nodes(node, [&](const NodeID adjacent_node) {
      graph_neighbours.push_back(adjacent_node);
    });

    compressed_graph.adjacent_nodes(node, [&](const NodeID adjacent_node) {
      compressed_graph_neighbours.push_back(adjacent_node);
    });

    if (graph_neighbours.size() != compressed_graph_neighbours.size()) {
      LOG_ERROR << "Node " << node << " has " << graph_neighbours.size()
                << " neighbours in the uncompressed graph but "
                << compressed_graph_neighbours.size() << " neighbours in the compressed graph!";
      return;
    }

    std::sort(graph_neighbours.begin(), graph_neighbours.end());
    std::sort(compressed_graph_neighbours.begin(), compressed_graph_neighbours.end());
    if (graph_neighbours != compressed_graph_neighbours) {
      LOG_ERROR << "The neighbourhood of node " << node
                << " in the compressed and uncompressed graph does not match!";
      return;
    }

    graph_neighbours.clear();
    compressed_graph_neighbours.clear();
  }
}

static void
expect_equal_neighbours(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  std::vector<EdgeID> graph_incident_edges;
  std::vector<NodeID> graph_adjacent_node;
  std::vector<EdgeID> compressed_graph_incident_edges;
  std::vector<NodeID> compressed_graph_adjacent_node;

  for (const NodeID node : graph.nodes()) {
    graph.neighbors(node, [&](const auto incident_edge, const auto adjacent_node) {
      graph_incident_edges.push_back(incident_edge);
      graph_adjacent_node.push_back(adjacent_node);
    });

    compressed_graph.neighbors(node, [&](const auto incident_edge, const auto adjacent_node) {
      compressed_graph_incident_edges.push_back(incident_edge);
      compressed_graph_adjacent_node.push_back(adjacent_node);
    });

    if (graph_incident_edges.size() != compressed_graph_incident_edges.size()) {
      LOG_ERROR << "Node " << node << " has " << graph_incident_edges.size()
                << " neighbours in the uncompressed graph but "
                << compressed_graph_incident_edges.size() << " neighbours in the compressed graph!";
      return;
    }

    std::sort(graph_incident_edges.begin(), graph_incident_edges.end());
    std::sort(graph_adjacent_node.begin(), graph_adjacent_node.end());
    std::sort(compressed_graph_incident_edges.begin(), compressed_graph_incident_edges.end());
    std::sort(compressed_graph_adjacent_node.begin(), compressed_graph_adjacent_node.end());

    if (graph_incident_edges != compressed_graph_incident_edges) {
      LOG_ERROR << "The incident edges of node " << node
                << " in the compressed and uncompressed graph do not match!";
      return;
    }

    if (graph_adjacent_node != compressed_graph_adjacent_node) {
      LOG_ERROR << "The adjacent nodes of node " << node
                << " in the compressed and uncompressed graph do not match!";
      return;
    }

    graph_incident_edges.clear();
    graph_adjacent_node.clear();
    compressed_graph_incident_edges.clear();
    compressed_graph_adjacent_node.clear();
  }
}

static void
expect_equal_pfor_neighbors(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  tbb::concurrent_vector<NodeID> graph_adjacent_node;
  tbb::concurrent_vector<NodeID> compressed_graph_adjacent_node;

  for (const NodeID node : graph.nodes()) {
    graph.pfor_neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        [&](const EdgeID e, const NodeID v) { graph_adjacent_node.push_back(v); }
    );

    compressed_graph.pfor_neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        [&](const EdgeID e, const NodeID v) { compressed_graph_adjacent_node.push_back(v); }
    );

    if (graph_adjacent_node.size() != compressed_graph_adjacent_node.size()) {
      LOG_ERROR << "Node " << node << " has " << graph_adjacent_node.size()
                << " adjacent nodes in the uncompressed graph but "
                << compressed_graph_adjacent_node.size()
                << " adjacent node in the compressed graph!";
      return;
    }

    std::sort(graph_adjacent_node.begin(), graph_adjacent_node.end());
    std::sort(compressed_graph_adjacent_node.begin(), compressed_graph_adjacent_node.end());

    if (graph_adjacent_node != compressed_graph_adjacent_node) {
      LOG_ERROR << "The adjacent nodes of node " << node
                << " in the compressed and uncompressed graph do not match!";
      return;
    }

    graph_adjacent_node.clear();
    compressed_graph_adjacent_node.clear();
  }
}

static void run_checks(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  LOG << "Checking if the graph operations are valid...";

  expect_equal_size(graph, compressed_graph);
  expect_equal_nodes(graph, compressed_graph);
  expect_equal_edges(graph, compressed_graph);
  expect_equal_degree(graph, compressed_graph);
  expect_equal_incident_edges(graph, compressed_graph);
  expect_equal_adjacent_nodes(graph, compressed_graph);
  expect_equal_neighbours(graph, compressed_graph);
  expect_equal_pfor_neighbors(graph, compressed_graph);
}

static void run_benchmark(CSRGraph graph, CompressedGraph compressed_graph) {
  LOG << "Running the benchmark...";

  START_HEAP_PROFILER("Uncompressed graph operations");
  TIMED_SCOPE("Uncompressed graph operations") {
    benchmark_degree(graph);
    benchmark_incident_edges(graph);
    benchmark_adjacent_nodes(graph);
    benchmark_neighbors(graph);
    benchmark_pfor_neighbors(graph);
  };
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Compressed graph operations");
  TIMED_SCOPE("Compressed graph operations") {
    benchmark_degree(compressed_graph);
    benchmark_incident_edges(compressed_graph);
    benchmark_adjacent_nodes(compressed_graph);
    benchmark_neighbors(compressed_graph);
    benchmark_pfor_neighbors(compressed_graph);
  };
  STOP_HEAP_PROFILER();

  Graph graph_csr(std::make_unique<CSRGraph>(std::move(graph)));
  START_HEAP_PROFILER("Uncompressed underlying graph operations");
  TIMED_SCOPE("Uncompressed underlying graph operations") {
    benchmark_degree(graph_csr);
    benchmark_incident_edges(graph_csr);
    benchmark_adjacent_nodes(graph_csr);
    benchmark_neighbors(graph_csr);
    benchmark_pfor_neighbors(graph_csr);
  };
  STOP_HEAP_PROFILER();

  Graph graph_compressed(std::make_unique<CompressedGraph>(std::move(compressed_graph)));
  START_HEAP_PROFILER("Compressed underlying graph operations");
  TIMED_SCOPE("Compressed underlying graph operations") {
    benchmark_degree(graph_compressed);
    benchmark_incident_edges(graph_compressed);
    benchmark_adjacent_nodes(graph_compressed);
    benchmark_neighbors(graph_compressed);
    benchmark_pfor_neighbors(graph_compressed);
  };
  STOP_HEAP_PROFILER();
}

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  std::string graph_filename;
  int num_threads = 1;
  bool enable_benchmarks = true;
  bool enable_checks = false;

  CLI::App app("Shared-memory graph compression benchmark");
  app.add_option("-G, --graph", graph_filename, "Graph file")->required();
  app.add_option("-t,--threads", num_threads, "Number of threads")
      ->check(CLI::NonNegativeNumber)
      ->default_val(num_threads);
  app.add_option("-b,--benchmark", enable_benchmarks, "Enable graph operations benchmark")
      ->default_val(enable_benchmarks);
  app.add_option("-c,--checks", enable_checks, "Enable compressed graph operations check")
      ->default_val(enable_checks);

  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  ENABLE_HEAP_PROFILER();
  GLOBAL_TIMER.reset();

  // Read input graph
  LOG << "Reading the input graph...";

  START_HEAP_PROFILER("CSR Graph Allocation");
  CSRGraph graph = TIMED_SCOPE("Read csr graph") {
    return io::metis::csr_read<false>(graph_filename);
  };
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Compressed Graph Allocation");
  CompressedGraph compressed_graph = TIMED_SCOPE("Read compressed graph") {
    return io::metis::compress_read<false>(graph_filename);
  };
  STOP_HEAP_PROFILER();

  // Capture graph statistics
  std::size_t csr_size = graph.raw_nodes().size() * sizeof(Graph::EdgeID) +
                         graph.raw_edges().size() * sizeof(Graph::NodeID);
  std::size_t compressed_size = compressed_graph.used_memory();
  std::size_t high_degree_count = compressed_graph.high_degree_count();
  std::size_t part_count = compressed_graph.part_count();
  std::size_t interval_count = compressed_graph.interval_count();

  // Run checks and benchmarks
  if (enable_checks) {
    run_checks(graph, compressed_graph);
  }

  if (enable_benchmarks) {
    run_benchmark(std::move(graph), std::move(compressed_graph));
  }

  STOP_TIMER();
  DISABLE_HEAP_PROFILER();

  // Print the result summary
  LOG;
  cio::print_delimiter("Result Summary");

  LOG << "Input graph has " << graph.n() << " vertices and " << graph.m()
      << " edges. Its density is " << ((graph.m()) / (float)(graph.n() * (graph.n() - 1))) << ".";
  LOG << "Node weights: " << (graph.is_node_weighted() ? "yes" : "no")
      << ", edge weights: " << (graph.is_edge_weighted() ? "yes" : "no");
  LOG;

  LOG << "The uncompressed graph uses " << to_megabytes(csr_size) << " mb (" << csr_size
      << " bytes).";
  LOG << "The compressed graph uses " << to_megabytes(compressed_size) << " mb (" << compressed_size
      << " bytes).";
  float compression_factor = csr_size / (float)compressed_size;
  LOG << "Thats a compression ratio of " << compression_factor << '.';
  LOG;

  LOG << high_degree_count << " (" << (high_degree_count / (float)graph.n())
      << "%) vertices have high degree.";
  LOG << part_count << " parts result from splitting the neighborhood of high degree nodes.";
  LOG << interval_count << " vertices/parts use interval encoding.";
  LOG;

  Timer::global().print_human_readable(std::cout);
  LOG;
  PRINT_HEAP_PROFILE(std::cout);

  return 0;
}
