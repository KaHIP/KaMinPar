/*******************************************************************************
 * Generic benchmark for the coarsening phase of the distributed algorithm.
 *
 * @file:   dist_coarsening_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   21.09.21
 ******************************************************************************/
// clang-format off
#include <kaminpar_cli/dkaminpar_arguments.h>
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/context.h"
#include "dkaminpar/presets.h"

#include "kaminpar/definitions.h"

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

  CLI::App app;
  app.add_option("-G", graph_filename);
  app.add_option("-t", ctx.parallel.num_threads);
  create_coarsening_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  omp_set_num_threads(ctx.parallel.num_threads);

  auto wrapper = load_graph(graph_filename);
  auto &graph = *wrapper.graph;
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  Coarsener coarsener(graph, ctx);
  const DistributedGraph *c_graph = &graph;

  while (c_graph->global_n() > ctx.partition.k * ctx.coarsening.contraction_limit) {
    const DistributedGraph *new_c_graph = coarsener.coarsen_once();
    if (new_c_graph == c_graph) {
      break;
    }
    c_graph = new_c_graph;

    LOG << "=> n=" << c_graph->global_n() << " m=" << c_graph->global_m()
        << " max_node_weight=" << c_graph->max_node_weight();
  }

  // Output statistics
  mpi::barrier(MPI_COMM_WORLD);
  LOG;
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_human_readable(std::cout);
  }
  LOG;

  return MPI_Finalize();
}
