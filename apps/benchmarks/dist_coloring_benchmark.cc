/*******************************************************************************
 * @file:   dist_coloring_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   25.01.2022
 * @brief:  Benchmark for the distributed vertex coloring algorithm.
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-format on

#include <mpi.h>
#include <omp.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/io.h"
#include "dkaminpar/timer.h"

#include "common/logger.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char *argv[]) {
  init_mpi(argc, argv);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  std::string graph_filename = "";
  int num_supersteps = 5;

  CLI::App app("Distributed Label Propagation Benchmark");
  app.add_option("-G,--graph", graph_filename, "Input graph")->required();
  app.add_option("-S,--num-supersteps", num_supersteps, "Number of supersteps")
      ->required();
  CLI11_PARSE(app, argc, argv);

  auto gc = init_parallelism(1);
  omp_set_num_threads(1);
  init_numa();

  /*****
   * Load graph
   */
  LOG << "Reading graph from " << graph_filename << " ...";
  DISABLE_TIMERS();
  auto graph = dist::io::read_graph(
      graph_filename, dist::io::DistributionType::NODE_BALANCED
  );
  ENABLE_TIMERS();
  LOG << "n=" << graph.global_n() << " m=" << graph.global_m();

  /****
   * Run label propagation
   */
  const auto coloring =
      compute_node_coloring_sequentially(graph, num_supersteps);
  const ColorID num_local_colors =
      *std::max_element(coloring.begin(), coloring.end()) + 1;
  const ColorID num_colors =
      mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());
  LOG << "num_colors=" << num_colors;

  std::vector<NodeID> color_sizes(num_colors);
  for (const NodeID u : graph.nodes()) {
    ++color_sizes[coloring[u]];
  }
  if (rank == 0) {
    MPI_Reduce(
        MPI_IN_PLACE,
        color_sizes.data(),
        asserting_cast<int>(num_colors),
        mpi::type::get<NodeID>(),
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
  } else {
    MPI_Reduce(
        color_sizes.data(),
        nullptr,
        asserting_cast<int>(num_colors),
        mpi::type::get<NodeID>(),
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
  }
  LOG << "color_sizes=" << color_sizes;

  /*****
   * Clean up and print timer tree
   */
  mpi::barrier(MPI_COMM_WORLD);
  STOP_TIMER();
  finalize_distributed_timer(GLOBAL_TIMER);
  if (rank == 0) {
    Timer::global().print_human_readable(std::cout);
  }
  return MPI_Finalize();
}
