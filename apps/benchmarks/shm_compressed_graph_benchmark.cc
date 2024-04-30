/*******************************************************************************
 * Graph compression benchmark for the shared-memory algorithm.
 *
 * @file:   shm_compressed_graph_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   12.11.2023
 ******************************************************************************/
#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/permutator.h"

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
  if (a.begin() == a.end()) {
    return b.begin() != b.end();
  }

  return a.begin() != b.begin() || a.end() != b.end();
};

// See https://github.com/google/benchmark/blob/main/include/benchmark/benchmark.h
template <class T> static inline void do_not_optimize(T value) {
  asm volatile("" : "+m"(value) : : "memory");
}

template <typename Graph> static void benchmark_degree(const Graph &graph) {
  SCOPED_TIMER("Degree");

  for (const auto node : graph.nodes()) {
    do_not_optimize(graph.degree(node));
  }
}

template <typename Graph> static void benchmark_incident_edges(const Graph &graph) {
  SCOPED_TIMER("Incident Edges");

  for (const auto node : graph.nodes()) {
    for (const auto incident_edge : graph.incident_edges(node)) {
      do_not_optimize(incident_edge);
    }
  }
}

template <typename Graph> static void benchmark_adjacent_nodes(const Graph &graph) {
  SCOPED_TIMER("Adjacent Nodes");

  for (const auto node : graph.nodes()) {
    graph.adjacent_nodes(node, [&](const auto adjacent_node) { do_not_optimize(adjacent_node); });
  }
}

template <typename Graph> static void benchmark_neighbors(const Graph &graph) {
  SCOPED_TIMER("Neighbors");

  for (const auto node : graph.nodes()) {
    graph.neighbors(node, [](const auto incident_edge, const auto adjacent_node) {
      do_not_optimize(incident_edge);
      do_not_optimize(adjacent_node);
    });
  }
}

template <typename Graph> static void benchmark_pfor_neighbors(const Graph &graph) {
  SCOPED_TIMER("Parallel For Neighbors");

  for (const auto node : graph.nodes()) {
    graph.pfor_neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        1000,
        [](const auto incident_edge, const auto adjacent_node) {
          do_not_optimize(incident_edge);
          do_not_optimize(adjacent_node);
        }
    );
  }
}

static void run_benchmark(CSRGraph graph, CompressedGraph compressed_graph) {
  LOG << "Running the benchmarks...";

  TIMED_SCOPE("Uncompressed graph operations") {
    benchmark_degree(graph);
    benchmark_incident_edges(graph);
    benchmark_adjacent_nodes(graph);
    benchmark_neighbors(graph);
    benchmark_pfor_neighbors(graph);
  };

  TIMED_SCOPE("Compressed graph operations") {
    benchmark_degree(compressed_graph);
    benchmark_incident_edges(compressed_graph);
    benchmark_adjacent_nodes(compressed_graph);
    benchmark_neighbors(compressed_graph);
    benchmark_pfor_neighbors(compressed_graph);
  };
}

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  std::string graph_filename;
  int num_threads = 1;
  bool enable_benchmarks = true;
  bool enable_checks = false;

  CLI::App app("Shared-memory graph compression benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-t,--threads", num_threads, "Number of threads")
      ->check(CLI::NonNegativeNumber)
      ->default_val(num_threads);
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
    return *io::metis::compress_read<false>(graph_filename);
  };
  STOP_HEAP_PROFILER();

  // Run benchmarks
  run_benchmark(std::move(graph), std::move(compressed_graph));

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

  Timer::global().print_human_readable(std::cout);
  LOG;
  PRINT_HEAP_PROFILE(std::cout);

  return 0;
}
