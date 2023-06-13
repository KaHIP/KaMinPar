/*******************************************************************************
 * @file:   dist_balancing_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Benchmark for the distributed balancing algorithm.
 ******************************************************************************/
// clang-format off
#include <kaminpar_cli/dkaminpar_arguments.h>
// clang-format on

#include <fstream>

#include <mpi.h>

#include "dist_io.h"

#include "dkaminpar/context.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/presets.h"

#include "kaminpar/definitions.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  Context ctx = create_default_context();
  std::string graph_filename;
  std::string partition_filename;

  // Remove default refiners
  ctx.refinement.algorithms.clear();

  CLI::App app;
  app.add_option("-G", graph_filename);
  app.add_option("-P", partition_filename);
  app.add_option("-e", ctx.partition.epsilon);
  app.add_option("-t", ctx.parallel.num_threads);
  create_refinement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);

  auto wrapper = load_partitioned_graph(graph_filename, partition_filename);
  auto &graph = *wrapper.graph;
  auto &p_graph = *wrapper.p_graph;

  ctx.partition.k = p_graph.k();
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  auto refiner = factory::create_refinement_algorithm(ctx);

  TIMED_SCOPE("Refiner") {
    TIMED_SCOPE("Initialization") {
      refiner->initialize(graph);
    };
    TIMED_SCOPE("Refinement") {
      refiner->refine(p_graph, ctx.partition);
    };
  };

  const auto cut_after = metrics::edge_cut(p_graph);
  const auto imbalance_after = metrics::imbalance(p_graph);
  LOG << "RESULT cut=" << cut_after << " imbalance=" << imbalance_after;
  mpi::barrier(MPI_COMM_WORLD);

  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_machine_readable(std::cout);
  }
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_human_readable(std::cout);
  }

  return MPI_Finalize();
}
