/*******************************************************************************
 * Benchmark for the distributed cluster contraction algorithm.
 *
 * @file:   dist_contraction_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   25.07.2022
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/dkaminpar_arguments.h>
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "kaminpar-dist/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/presets.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

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
  app.add_option("-G,--graph", graph_filename)->check(CLI::ExistingFile);
  app.add_option("-C,--clustering", clustering_filename)->check(CLI::ExistingFile);
  app.add_option("-t,--threads", ctx.parallel.num_threads);
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
  print_graph_summary(result.graph);

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
