/*******************************************************************************
 * Benchmark for the move sets construction.
 *
 * @file:   dist_move_sets_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   04.07.2023
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
#include "kaminpar-dist/refinement/balancer/clusters.h"

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
  std::string partition_filename;
  NodeWeight max_move_set_size = 64;
  ClusterStrategy strategy = ClusterStrategy::GREEDY_BATCH_PREFIX;

  // Remove default refiners
  ctx.refinement.algorithms.clear();

  CLI::App app;
  app.add_option("-G,--graph", graph_filename);
  app.add_option("-P,--partition", partition_filename);
  app.add_option("-e,--epsilon", ctx.partition.epsilon);
  app.add_option("-t,--threads", ctx.parallel.num_threads);
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

  const Clusters sets =
      build_clusters(strategy, p_graph, ctx, ctx.partition, max_move_set_size, {});

  LOG << "Number of move sets: " << sets.num_clusters();

  NodeID max_size = 0;
  NodeID min_size = graph.global_n();
  NodeID sum = 0;
  NodeID count = sets.num_clusters();
  std::vector<NodeID> set_sizes;
  for (NodeID set = 0; set < sets.num_clusters(); ++set) {
    set_sizes.push_back(sets.size(set));
    max_size = std::max<NodeID>(max_size, sets.size(set));
    min_size = std::min<NodeID>(min_size, sets.size(set));
    sum += sets.size(set);
  }

  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  std::vector<int> counts(size);
  MPI_Allgather(&count, 1, mpi::type::get<NodeID>(), counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> displs(size + 1);
  std::partial_sum(counts.begin(), counts.end(), displs.begin() + 1);

  MPI_Allreduce(MPI_IN_PLACE, &max_size, 1, mpi::type::get<NodeID>(), MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &min_size, 1, mpi::type::get<NodeID>(), MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, mpi::type::get<NodeID>(), MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, mpi::type::get<NodeID>(), MPI_SUM, MPI_COMM_WORLD);

  std::vector<NodeID> global_set_sizes(count);
  MPI_Allgatherv(
      set_sizes.data(),
      set_sizes.size(),
      mpi::type::get<NodeID>(),
      global_set_sizes.data(),
      counts.data(),
      displs.data(),
      mpi::type::get<NodeID>(),
      MPI_COMM_WORLD
  );
  std::sort(global_set_sizes.begin(), global_set_sizes.end());

  LOG << "Max: " << max_size << ", avg: " << 1.0 * sum / count << ", min: " << min_size
      << ", sum: " << sum;
  LOG << "0.1-quantile: " << global_set_sizes[global_set_sizes.size() / 10]
      << ", 0.25-quantile: " << global_set_sizes[global_set_sizes.size() / 4]
      << ", median: " << global_set_sizes[global_set_sizes.size() / 2]
      << ", 0.75-quantile: " << global_set_sizes[3.0 * global_set_sizes.size() / 4]
      << ", 0.9-quantile: " << global_set_sizes[9.0 * global_set_sizes.size() / 10];

  mpi::barrier(MPI_COMM_WORLD);
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_machine_readable(std::cout);
  }
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    Timer::global().print_human_readable(std::cout);
  }

  return MPI_Finalize();
}
