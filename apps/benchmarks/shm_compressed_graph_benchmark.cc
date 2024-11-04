/*******************************************************************************
 * Compressed graph benchmark for the shared-memory algorithm.
 *
 * @file:   shm_compressed_graph_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   12.11.2023
 ******************************************************************************/
#include <limits>

#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/graphutils/parallel_compressed_graph_builder.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::io;

namespace {

template <typename T> static bool operator!=(const IotaRange<T> &a, const IotaRange<T> &b) {
  if (a.begin() == a.end()) {
    return b.begin() != b.end();
  }

  return a.begin() != b.begin() || a.end() != b.end();
}

// See https://github.com/google/benchmark/blob/main/include/benchmark/benchmark.h
template <class T> void do_not_optimize(T value) {
  asm volatile("" : "+m"(value) : : "memory");
}

template <typename Graph> void benchmark_degree(const Graph &graph) {
  SCOPED_TIMER("Degree");

  for (const auto node : graph.nodes()) {
    do_not_optimize(graph.degree(node));
  }
}

template <typename Graph> void benchmark_incident_edges(const Graph &graph) {
  SCOPED_TIMER("Incident Edges");

  for (const auto node : graph.nodes()) {
    for (const auto incident_edge : graph.incident_edges(node)) {
      do_not_optimize(incident_edge);
    }
  }
}

template <typename Graph> void benchmark_adjacent_nodes(const Graph &graph) {
  SCOPED_TIMER("Adjacent Nodes");

  for (const auto node : graph.nodes()) {
    graph.adjacent_nodes(node, [&](const auto adjacent_node) { do_not_optimize(adjacent_node); });
  }
}

template <typename Graph> void benchmark_weighted_adjacent_nodes(const Graph &graph) {
  SCOPED_TIMER("Adjacent Nodes with Edge Weights");

  for (const auto node : graph.nodes()) {
    graph.adjacent_nodes(node, [&](const auto adjacent_node, const auto edge_weight) {
      do_not_optimize(adjacent_node);
      do_not_optimize(edge_weight);
    });
  }
}

template <typename Graph> void benchmark_neighbors(const Graph &graph) {
  SCOPED_TIMER("Neighbors");

  for (const auto node : graph.nodes()) {
    graph.neighbors(node, [](const auto incident_edge, const auto adjacent_node) {
      do_not_optimize(incident_edge);
      do_not_optimize(adjacent_node);
    });
  }
}

template <typename Graph> void benchmark_weighted_neighbors(const Graph &graph) {
  SCOPED_TIMER("Neighbors with Edge Weights");

  for (const auto node : graph.nodes()) {
    graph.neighbors(
        node,
        [](const auto incident_edge, const auto adjacent_node, const auto edge_weight) {
          do_not_optimize(incident_edge);
          do_not_optimize(adjacent_node);
          do_not_optimize(edge_weight);
        }
    );
  }
}

template <typename Graph> void benchmark_neighbors_limit(const Graph &graph) {
  SCOPED_TIMER("Neighbors (with limit)");

  for (const auto node : graph.nodes()) {
    graph.neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        [](const auto incident_edge, const auto adjacent_node) {
          do_not_optimize(incident_edge);
          do_not_optimize(adjacent_node);
        }
    );
  }
}

template <typename Graph> void benchmark_weighted_neighbors_limit(const Graph &graph) {
  SCOPED_TIMER("Neighbors with Edge Weights (with limit)");

  for (const auto node : graph.nodes()) {
    graph.neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        [](const auto incident_edge, const auto adjacent_node, const auto edge_weight) {
          do_not_optimize(incident_edge);
          do_not_optimize(adjacent_node);
          do_not_optimize(edge_weight);
        }
    );
  }
}

template <typename Graph> void benchmark_pfor_neighbors(const Graph &graph) {
  SCOPED_TIMER("Parallel For Neighbors");

  for (const auto node : graph.nodes()) {
    graph.pfor_neighbors(
        node,
        std::numeric_limits<NodeID>::max(),
        1000,
        [](const auto incident_edge, const auto adjacent_node, const auto edge_weight) {
          do_not_optimize(incident_edge);
          do_not_optimize(adjacent_node);
          do_not_optimize(edge_weight);
        }
    );
  }
}

void run_benchmark(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  TIMED_SCOPE("Uncompressed graph operations") {
    benchmark_degree(graph);
    benchmark_incident_edges(graph);
    benchmark_adjacent_nodes(graph);
    benchmark_weighted_adjacent_nodes(graph);
    benchmark_neighbors(graph);
    benchmark_weighted_neighbors(graph);
    benchmark_neighbors_limit(graph);
    benchmark_weighted_neighbors_limit(graph);
    benchmark_pfor_neighbors(graph);
  };

  TIMED_SCOPE("Compressed graph operations") {
    benchmark_degree(compressed_graph);
    benchmark_incident_edges(compressed_graph);
    benchmark_adjacent_nodes(compressed_graph);
    benchmark_weighted_adjacent_nodes(compressed_graph);
    benchmark_neighbors(compressed_graph);
    benchmark_weighted_neighbors(compressed_graph);
    benchmark_neighbors_limit(compressed_graph);
    benchmark_weighted_neighbors_limit(compressed_graph);
    benchmark_pfor_neighbors(compressed_graph);
  };
}

} // namespace

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  std::string graph_filename;
  GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  int num_threads = 1;

  CLI::App app("Shared-memory graph compression benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)");
  app.add_option("-t,--threads", num_threads, "Number of threads")
      ->check(CLI::NonNegativeNumber)
      ->default_val(num_threads);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  // Read input graph
  LOG << "Reading the input graph...";
  CSRGraph graph = io::csr_read(graph_filename, graph_file_format, NodeOrdering::NATURAL);

  LOG << "Compressing the input graph...";
  CompressedGraph compressed_graph = parallel_compress(graph);

  // Run benchmarks
  LOG << "Running the benchmarks...";
  GLOBAL_TIMER.reset();
  run_benchmark(graph, compressed_graph);
  STOP_TIMER();

  // Print the result summary
  LOG;
  cio::print_delimiter("Result Summary");

  LOG << "Input graph has " << graph.n() << " vertices and " << graph.m()
      << " edges. Its density is " << ((graph.m()) / (float)(graph.n() * (graph.n() - 1))) << ".";
  LOG << "Node weights: " << (graph.is_node_weighted() ? "yes" : "no")
      << ", edge weights: " << (graph.is_edge_weighted() ? "yes" : "no");
  LOG;

  Timer::global().print_human_readable(std::cout);

  return EXIT_SUCCESS;
}
