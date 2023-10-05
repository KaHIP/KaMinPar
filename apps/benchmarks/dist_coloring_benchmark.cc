/*******************************************************************************
 * Benchmark for the distributed node coloring algorithm.
 *
 * @file:   dist_coloring_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   25.01.2022
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/dkaminpar_arguments.h>
// clang-format on

#include <mpi.h>
#include <omp.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-dist/algorithms/greedy_node_coloring.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

#include "apps/benchmarks/dist_io.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  Context ctx = create_default_context();
  std::string graph_filename = "";
  int num_supersteps = 5;

  CLI::App app("Distributed Label Propagation Benchmark");
  app.add_option("-G,--graph", graph_filename, "Input graph")->check(CLI::ExistingFile)->required();
  app.add_option("-S,--num-supersteps", num_supersteps, "Number of supersteps")->required();
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  omp_set_num_threads(ctx.parallel.num_threads);

  auto wrapper = load_graph(graph_filename);
  auto &graph = *wrapper.graph;
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  const auto coloring = compute_node_coloring_sequentially(graph, num_supersteps);
  const ColorID num_local_colors = *std::max_element(coloring.begin(), coloring.end()) + 1;
  const ColorID num_colors = mpi::allreduce(num_local_colors, MPI_MAX, graph.communicator());
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
