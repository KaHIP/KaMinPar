/*******************************************************************************
 * @file:   shm_refinement_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   20.04.2023
 * @brief:  Isolated benchmark for refinement algorithms.
 ******************************************************************************/
// clang-format off
#include <kaminpar_cli/kaminpar_arguments.h>
// clang-format on

#include <common/timer.h>
#include <kaminpar/factories.h>
#include <mpi.h>

#include "io.h"

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

  CLI::App app("Shared-memory FM benchmark");
  app.add_option("graph", graph_filename, "Graph file")->required();
  app.add_option("partition", partition_filename, "Partition file")->required();
  create_refinement_options(&app, ctx);
  create_partitioning_options(&app, ctx);
  create_kway_fm_refinement_options(&app, ctx);
  create_lp_refinement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  if (ctx.refinement.algorithms.empty()) {
    std::cerr << "Error: no refinement algorithm selected, use --r-algorithms "
                 "to specify which algorithm to benchmark."
              << std::endl;
    return MPI_Finalize();
  }

  // Load input graph
  auto input = load_partitioned_shm_graph(graph_filename, partition_filename);
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
            << " imbalance=" << metrics::imbalance(*input.p_graph);

  STOP_TIMER(); // Stop root timer
  Timer::global().print_human_readable(std::cout);

  return MPI_Finalize();
}
