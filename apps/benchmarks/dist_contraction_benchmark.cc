/*******************************************************************************
 * @file:   dcontraction_benchmark.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.05.2022
 * @brief:  Benchmark for distributed graph contraction.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "common/CLI11.h"
#include "dkaminpar/definitions.h"
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/graphutils/rearrangement.h"
#include "dkaminpar/io.h"
#include "dkaminpar/mpi/wrapper.h"
#include "dkaminpar/presets.h"

#include "kaminpar/definitions.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/benchmarks/dist_benchmarks_common.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

auto load_clustering(const std::string &filename, const NodeID local_n) {
  return dist::io::partition::read<
      scalable_vector<parallel::Atomic<GlobalNodeID>>>(filename, local_n);
}

int main(int argc, char *argv[]) {
  init_mpi(argc, argv);

  Context ctx = create_default_context();

  std::string graph_filename = "";
  std::string clustering_filename = "";

  CLI::App app("Distributed Graph Contraction Benchmark");
  app.add_option("-G,--graph", graph_filename, "Input graph")->required();
  app.add_option(
      "-C,--clustering", clustering_filename, "Name of the clustering file."
  );
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  CLI11_PARSE(app, argc, argv);

  auto gc = init(ctx, argc, argv);

  // Load data
  const auto graph = load_graph(graph_filename);
  ctx.setup(graph);
  const auto clustering = load_clustering(clustering_filename, graph.n());

  // Compute coarse graph
  START_TIMER("Contraction");
  const auto result = contract_global_clustering(
      graph, clustering, ctx.coarsening.global_contraction_algorithm
  );
  STOP_TIMER();

  LOG << "Coarse graph:";
  graph::print_summary(c_graph);

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
