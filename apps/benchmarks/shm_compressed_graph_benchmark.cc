/*******************************************************************************
 * Graph compression benchmark for the shared-memory algorithm.
 *
 * @file:   shm_compressed_graph_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   12.11.2023
 ******************************************************************************/
#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"
#include "kaminpar-common/variable_length_codec.h"

#include "apps/io/shm_io.cc"

using namespace kaminpar;
using namespace kaminpar::shm;

static std::string to_megabytes(std::size_t bytes) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << (bytes / (float)(1024 * 1024));
  return stream.str();
}

template <typename Graph> void benchmark_degree(const Graph &graph) {
  SCOPED_TIMER("Degree");

  volatile std::size_t total_degree = 0;
  for (const auto node : graph.nodes()) {
    for (std::size_t i = 0; i < 1000; ++i) {
      total_degree += graph.degree(node);
    }
  }
}

template <typename Graph> void benchmark_adjacent_nodes(const Graph &graph) {
  SCOPED_TIMER("Adjacent nodes");

  volatile std::size_t count = 0;
  for (const auto node : graph.nodes()) {
    for (std::size_t i = 0; i < 100; ++i) {
      for (const auto adjacent_node : graph.adjacent_nodes(node)) {
        count++;
      }
    }
  }
}

void expect_equal_degree(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  for (const auto node : graph.nodes()) {
    if (graph.degree(node) != compressed_graph.degree(node)) {
      LOG << "The node " << node << " has degree " << compressed_graph.degree(node)
          << " in the compressed graph and degree" << graph.degree(node)
          << " in the uncompressed graph!";
      return;
    }
  }
}

void expect_compressed_graph_eq(const CSRGraph &graph, const CompressedGraph &compressed_graph) {
  if (graph.n() != compressed_graph.n()) {
    LOG << "The uncompressed graph has " << graph.n() << " nodes and the compressed graph has "
        << compressed_graph.n() << " nodes!";
    return;
  }

  if (graph.m() != compressed_graph.m()) {
    LOG << "The uncompressed graph has " << graph.m() << " edges and the compressed graph has "
        << compressed_graph.m() << " edges!";
    return;
  }

  for (const NodeID node : graph.nodes()) {
    std::vector<NodeID> graph_neighbours;
    std::vector<NodeID> compressed_graph_neighbours;

    for (const NodeID adjacent_node : graph.adjacent_nodes(node)) {
      graph_neighbours.push_back(adjacent_node);
    }

    for (const NodeID adjacent_node : compressed_graph.adjacent_nodes(node)) {
      compressed_graph_neighbours.push_back(adjacent_node);
    }

    if (graph_neighbours.size() != compressed_graph_neighbours.size()) {
      LOG << "Node " << node << " has " << graph_neighbours.size()
          << " neighbours in the uncompressed graph but " << compressed_graph_neighbours.size()
          << " neighbours in the compressed graph!";
      return;
    }

    std::sort(graph_neighbours.begin(), graph_neighbours.end());
    std::sort(compressed_graph_neighbours.begin(), compressed_graph_neighbours.end());
    if (graph_neighbours != compressed_graph_neighbours) {
      LOG << "The neighbourhood of node " << node
          << " int graph and the compressed graph does not match!";
    }
  }
}

struct GraphStats {
  std::size_t used_memory;
  std::size_t interval_count;
};

GraphStats run_benchmark(const CSRGraph &graph, bool benchmarks, bool checks) {
  auto compressed_graph = CompressedGraph::compress(graph);

  if (benchmarks) {
    LOG << "Running the benchmark...";

    START_HEAP_PROFILER("Compressed graph operations");
    TIMED_SCOPE("Compressed graph operations") {
      benchmark_degree(compressed_graph);
      benchmark_adjacent_nodes(compressed_graph);
    };
    STOP_HEAP_PROFILER();

    START_HEAP_PROFILER("Uncompressed graph operations");
    TIMED_SCOPE("Uncompressed graph operations") {
      benchmark_degree(graph);
      benchmark_adjacent_nodes(graph);
    };
    STOP_HEAP_PROFILER();
  }

  if (checks) {
    LOG << "Checking if the graph operations are valid...";
    expect_equal_degree(graph, compressed_graph);
    expect_compressed_graph_eq(graph, compressed_graph);
  }

  return {compressed_graph.used_memory(), compressed_graph.interval_count()};
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

  // Read input graph
  LOG << "Reading the input graph...";
  StaticArray<EdgeID> xadj;
  StaticArray<NodeID> adjncy;
  StaticArray<NodeWeight> vwgt;
  StaticArray<EdgeWeight> adjwgt;

  shm::io::metis::read<false>(graph_filename, xadj, adjncy, vwgt, adjwgt);

  const NodeID n = static_cast<NodeID>(xadj.size() - 1);
  const EdgeID m = xadj[n];

  StaticArray<EdgeID> nodes(xadj.data(), n + 1);
  StaticArray<NodeID> edges(adjncy.data(), m);
  StaticArray<NodeWeight> node_weights =
      (vwgt.empty()) ? StaticArray<NodeWeight>(0) : StaticArray<NodeWeight>(vwgt.data(), n);
  StaticArray<EdgeWeight> edge_weights =
      (adjwgt.empty()) ? StaticArray<EdgeWeight>(0) : StaticArray<EdgeWeight>(adjwgt.data(), m);

  CSRGraph graph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)
  );

  // Run the benchmark
  ENABLE_HEAP_PROFILER();
  GLOBAL_TIMER.reset();

  LOG << "Compressing the input graph...";
  using VarLengthCodec = VarIntCodec;

  GraphStats stats = run_benchmark(graph, enable_benchmarks, enable_checks);

  STOP_TIMER();
  DISABLE_HEAP_PROFILER();

  // Print the result summary
  LOG;
  cio::print_delimiter("Result Summary");

  LOG << "Input graph has " << graph.n() << " vertices and " << graph.m()
      << " edges. Its density is " << ((graph.m()) / (float)(graph.n() * (graph.n() - 1))) << ".";
  LOG;
  std::size_t graph_size = graph.raw_nodes().size() * sizeof(Graph::EdgeID) +
                           graph.raw_edges().size() * sizeof(Graph::NodeID);
  LOG << "The uncompressed graph uses " << to_megabytes(graph_size) << " mb (" << graph_size
      << " bytes).";

  std::size_t compressed_size = stats.used_memory;
  LOG << "The compressed graph uses " << to_megabytes(compressed_size) << " mb (" << compressed_size
      << " bytes).";

  float compression_factor = graph_size / (float)compressed_size;
  LOG << "Thats a compression ratio of " << compression_factor << '.';
  LOG;

  std::size_t interval_count = stats.interval_count;
  LOG << interval_count << " (" << (interval_count / (float)graph.n())
      << "%) vertices use interval encoding.";
  LOG;

  Timer::global().print_human_readable(std::cout);
  LOG;
  PRINT_HEAP_PROFILE(std::cout);

  return 0;
}
