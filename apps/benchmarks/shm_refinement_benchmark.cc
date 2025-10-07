/*******************************************************************************
 * Generic refinement benchmark for the shared-memory algorithm.
 *
 * @file:   shm_refinement_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   08.04.2025
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include "kaminpar-io/kaminpar_io.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  ctx.parallel.num_threads = tbb::this_task_arena::max_concurrency();

  double epsilon = 0.03;
  int seed = 0;

  std::string graph_filename;
  std::string partition_filename;
  io::GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;

  CLI::App app("Shared-memory refinement benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-P,--partition", partition_filename, "Partition file")->required();
  app.add_option(
         "-e,--epsilon",
         epsilon,
         "Maximum allowed imbalance, e.g., 0.03 for 3%. Must be strictly positive."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(
          CLI::CheckedTransformer(
              std::unordered_map<std::string, io::GraphFileFormat>{
                  {"metis", io::GraphFileFormat::METIS},
                  {"parhip", io::GraphFileFormat::PARHIP},
                  {"compressed", io::GraphFileFormat::COMPRESSED},
              },
              CLI::ignore_case
          )
      );
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  app.add_option("-s,--seed", seed, "Seed for random number generation.")->default_val(seed);

  create_refinement_options(&app, ctx);
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  KaMinPar::reseed(seed);

  auto graph = io::read_graph(graph_filename, graph_file_format, ctx.compression.enabled);
  if (!graph) {
    LOG_ERROR << "Failed to read the input graph";
    return EXIT_FAILURE;
  }

  std::vector<BlockID> partition = io::read_partition(partition_filename);
  const BlockID k = std::unordered_set<BlockID>(partition.begin(), partition.end()).size();

  if (partition.size() != graph->n()) {
    LOG_ERROR << "Partition size (" << partition.size()
              << ") does not match number of nodes in graph (" << graph->n() << ")";
    return EXIT_FAILURE;
  }

  PartitionedGraph p_graph(*graph, k, StaticArray<BlockID>(partition.size(), partition.data()));
  ctx.compression.setup(*graph);
  ctx.partition.setup(*graph, k, epsilon);

  cio::print_kaminpar_banner();
  cio::print_build_identifier();
  cio::print_build_datatypes<NodeID, EdgeID, NodeWeight, EdgeWeight>();

  cio::print_delimiter("Input Summary");
  std::cout << "Threads:                      " << ctx.parallel.num_threads << "\n";
  std::cout << "Seed:                         " << seed << "\n";
  std::cout << "Graph:                        " << ctx.debug.graph_name
            << " [node ordering: " << ctx.node_ordering << "] [edge ordering: " << ctx.edge_ordering
            << "]\n";
  std::cout << ctx.partition;

  cio::print_delimiter("Refinement", '-');
  std::cout << ctx.refinement;

  cio::print_delimiter("Refinement");

  const auto print_statistics = [&]() {
    const EdgeWeight cut = metrics::edge_cut(p_graph);
    const double imbalance = metrics::imbalance(p_graph);
    const bool feasible = metrics::is_feasible(p_graph, ctx.partition);
    LOG << "Cut=" << cut << ", Imbalance=" << imbalance
        << ", Feasible=" << (feasible ? "Yes" : "No") << ", k=" << p_graph.k();
  };
  print_statistics();
  LOG;

  std::unique_ptr<Refiner> refiner = factory::create_refiner(ctx);

  ENABLE_HEAP_PROFILER();
  GLOBAL_TIMER.reset();

  refiner->refine(p_graph, ctx.partition);

  STOP_TIMER();
  DISABLE_HEAP_PROFILER();

  cio::print_delimiter("Result Summary");
  print_statistics();
  LOG;

  Timer::global().print_human_readable(std::cout);
  LOG;

  heap_profiler::HeapProfiler::global().set_experiment_summary_options();
  PRINT_HEAP_PROFILE(std::cout);

  return EXIT_SUCCESS;
}
