/*******************************************************************************
 * Benchmark for the distributed cluster contraction algorithm.
 *
 * @file:   dist_contraction_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   25.07.2022
 ******************************************************************************/
// clang-format off
#include <kaminpar_cli/dkaminpar_arguments.h>
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/coarsening/contraction/cluster_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/presets.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/benchmarks/dist_io.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  Context ctx = create_default_context();
  std::string graph_filename;
  std::string clustering_filename;

  // Remove default refiners
  ctx.refinement.algorithms.clear();

  CLI::App app;
  app.add_option("-G", graph_filename)->check(CLI::ExistingFile);
  app.add_option("-C", clustering_filename)->check(CLI::ExistingFile);
  app.add_option("-t", ctx.parallel.num_threads);
  create_coarsening_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  omp_set_num_threads(ctx.parallel.num_threads);

  auto wrapper = load_graph(graph_filename);
  auto &graph = *wrapper.graph;
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  GlobalClustering clustering =
      load_node_property_vector<NoinitVector<GlobalNodeID>>(graph, clustering_filename);

  // Compute coarse graph
  START_TIMER("Contraction");
  const auto result = contract_clustering(graph, clustering, ctx.coarsening);
  STOP_TIMER();

  LOG << "Coarse graph:";
  graph::print_summary(result.graph);

  // Output statistics
  mpi::barrier(MPI_COMM_WORLD);
  LOG;
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_human_readable(std::cout);
  }
  LOG;

  MPI_Finalize();
  return 0;
}
