/*******************************************************************************
 * Benchmark for the move sets construction.
 *
 * @file:   dist_move_sets_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   04.07.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar_cli/dkaminpar_arguments.h>
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/context.h"
#include "dkaminpar/context_io.h"
#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/presets.h"
#include "dkaminpar/refinement/balancer/move_sets.h"

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
  std::string partition_filename;
  NodeWeight max_move_set_size = 64;
  MoveSetStrategy strategy = MoveSetStrategy::GREEDY_BATCH_PREFIX;

  // Remove default refiners
  ctx.refinement.algorithms.clear();

  CLI::App app;
  app.add_option("-G", graph_filename);
  app.add_option("-P", partition_filename);
  app.add_option("-e", ctx.partition.epsilon);
  app.add_option("-t", ctx.parallel.num_threads);
  app.add_option("--size", max_move_set_size);
  app.add_option("--strategy", strategy)
      ->transform(CLI::CheckedTransformer(get_move_set_strategies()));
  create_refinement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  omp_set_num_threads(ctx.parallel.num_threads);

  auto wrapper = load_partitioned_graph(graph_filename, partition_filename);
  auto &graph = *wrapper.graph;
  auto &p_graph = *wrapper.p_graph;

  LOG << "Number of nodes: " << graph.global_n();
  LOG << "Number of edges: " << graph.global_m();

  ctx.partition.k = p_graph.k();
  ctx.partition.graph = std::make_unique<GraphContext>(graph, ctx.partition);

  const MoveSets sets =
      build_move_sets(strategy, p_graph, ctx, ctx.partition, max_move_set_size, {});

  LOG << "Number of move sets: " << sets.num_move_sets();

  NodeID max_size = 0;
  NodeID min_size = graph.global_n();
  NodeID sum = 0;
  std::vector<NodeID> set_sizes;
  for (NodeID set = 0; set < sets.num_move_sets(); ++set) {
    set_sizes.push_back(sets.size(set));
    max_size = std::max<NodeID>(max_size, sets.size(set));
    min_size = std::min<NodeID>(min_size, sets.size(set));
    sum += sets.size(set);
    if (sum > graph.global_n()) {
      LOG << "Sum is larger than the number of nodes!" << sum;
    }
  }

  std::sort(set_sizes.begin(), set_sizes.end());

  LOG << "Max: " << max_size << ", avg: " << 1.0 * graph.global_n() / sets.num_move_sets()
      << ", min: " << min_size << ", sum: " << sum;
  LOG << "0.1-quantile: " << set_sizes[set_sizes.size() / 10]
      << ", 0.25-quantile: " << set_sizes[set_sizes.size() / 4]
      << ", median: " << set_sizes[set_sizes.size() / 2]
      << ", 0.75-quantile: " << set_sizes[3.0 * set_sizes.size() / 4]
      << ", 0.9-quantile: " << set_sizes[9.0 * set_sizes.size() / 10];

  mpi::barrier(MPI_COMM_WORLD);
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_machine_readable(std::cout);
  }
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_human_readable(std::cout);
  }

  return MPI_Finalize();
}

