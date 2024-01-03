/*******************************************************************************
 * Generic refinement benchmark for the distributed algorithm.
 *
 * @file:   dist_balancing_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/dkaminpar_arguments.h>
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/context_io.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/presets.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

#include "apps/benchmarks/dist_io.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    cio::print_dkaminpar_banner();
  }

  Context ctx = create_default_context();
  std::string graph_filename;
  std::string partition_filename;

  // Remove default refiners
  ctx.refinement.algorithms.clear();

  CLI::App app;
  app.add_option("-G,--graph", graph_filename);
  app.add_option("-P,--partition", partition_filename);
  app.add_option("-e,--epsilon", ctx.partition.epsilon);
  app.add_option("-t,--threads", ctx.parallel.num_threads);
  create_refinement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    cio::print_build_identifier();
    cio::print_delimiter("Configuration", '-');
    print(ctx.refinement, ctx.parallel, std::cout);
    cio::print_delimiter();
  }

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  omp_set_num_threads(ctx.parallel.num_threads);

  auto wrapper = load_partitioned_graph(graph_filename, partition_filename);
  auto &graph = *wrapper.graph;
  auto &p_graph = *wrapper.p_graph;

  ctx.partition.k = p_graph.k();
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    LOG << "Block weights of the input partition:";
    LOG << logger::Table{8} << p_graph.block_weights();
  }

  auto refiner_factory = factory::create_refiner(ctx);
  auto refiner = refiner_factory->create(p_graph, ctx.partition);

  TIMED_SCOPE("Refiner") {
    TIMED_SCOPE("Initialization") {
      refiner->initialize();
    };
    TIMED_SCOPE("Refinement") {
      refiner->refine();
    };
  };

  const auto cut_after = metrics::edge_cut(p_graph);
  const auto imbalance_after = metrics::imbalance(p_graph);

  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    LOG << "RESULT cut=" << cut_after << " imbalance=" << imbalance_after;
    LOG << "Block weights of the resulting partition:";
    LOG << logger::Table{8} << p_graph.block_weights();
  }

  mpi::barrier(MPI_COMM_WORLD);

  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_machine_readable(std::cout);
  }
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_human_readable(std::cout);
  }

  return MPI_Finalize();
}
