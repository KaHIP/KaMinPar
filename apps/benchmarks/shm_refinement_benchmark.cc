/*******************************************************************************
 * Generic refinement benchmark for the shared-memory algorithm.
 *
 * @file:   shm_refinement_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   20.04.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <mpi.h>
#include <tbb/global_control.h>

#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/timer.h"

#include "apps/benchmarks/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Create context with no preselected refinement algorithms
  Context ctx = create_default_context();
  ctx.refinement.algorithms.clear();

  // Parse CLI arguments
  std::string graph_filename;
  std::string partition_filename;
  bool is_sorted = false;
  int num_threads = 1;

  CLI::App app("Shared-memory FM benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-P,--partition", partition_filename, "Partition file")->required();
  app.add_option("-t,--threads", num_threads, "Number of threads");
  app.add_flag("--deg-sorted-input", is_sorted)->capture_default_str();
  create_refinement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  if (ctx.refinement.algorithms.empty()) {
    std::cerr << "Error: no refinement algorithm selected, use --r-algorithms "
                 "to specify which algorithm to benchmark."
              << std::endl;
    return MPI_Finalize();
  }

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  // Load input graph
  {
    auto input = load_partitioned_graph(graph_filename, partition_filename, is_sorted);

    ctx.partition.k = input.p_graph->k();
    ctx.parallel.num_threads = num_threads;
    ctx.setup(*input.graph);

    std::cout << "Running refinement algorithm ..." << std::endl;

    START_TIMER("Benchmark");
    START_TIMER("Allocation");
    auto refiner = factory::create_refiner(ctx);
    STOP_TIMER();
    START_TIMER("Initialize");
    refiner->initialize(*input.p_graph);
    STOP_TIMER();
    START_TIMER("Refinement");
    refiner->refine(*input.p_graph, ctx.partition);
    STOP_TIMER();
    STOP_TIMER();

    std::cout << "RESULT cut=" << metrics::edge_cut(*input.p_graph)
              << " imbalance=" << metrics::imbalance(*input.p_graph) << std::endl;
  }

  STOP_TIMER(); // Stop root timer
  Timer::global().print_human_readable(std::cout);

  return MPI_Finalize();
}
