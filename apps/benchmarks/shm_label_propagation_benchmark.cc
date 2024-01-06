/*******************************************************************************
 * Generic label propagation benchmark for the shared-memory algorithm.
 *
 * @file:   shm_label_propagation_benchmark.cc
 * @author: Daniel Salwasser
 * @date:   13.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/coarsening/lp_clustering.h"
#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/partition_utils.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  // Create context
  Context ctx = create_default_context();

  // Parse CLI arguments
  std::string graph_filename;
  int seed = 0;

  CLI::App app("Shared-memory LP benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  app.add_option("-s,--seed", seed, "Seed for random number generation.")->default_val(seed);
  app.add_option("-k,--k", ctx.partition.k, "Number of blocks in the partition.")->required();
  app.add_option(
         "-e,--epsilon",
         ctx.partition.epsilon,
         "Maximum allowed imbalance, e.g. 0.03 for 3%. Must be strictly positive."
  )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();
  create_lp_coarsening_options(&app, ctx);
  create_partitioning_rearrangement_options(&app, ctx);
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);
  Random::reseed(seed);

  Graph graph = io::read(
      graph_filename,
      ctx.compression.enabled,
      ctx.node_ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS,
      false
  );
  ctx.setup(graph);

  if (ctx.node_ordering == NodeOrdering::DEGREE_BUCKETS) {
    const double original_epsilon = ctx.partition.epsilon;
    graph = graph::rearrange_by_degree_buckets(ctx, std::move(graph));
    graph::integrate_isolated_nodes(graph, original_epsilon, ctx);
  }

  const NodeWeight max_cluster_weight =
      compute_max_cluster_weight(ctx.coarsening, graph, ctx.partition);

  LPClustering lp_clustering(graph.n(), ctx.coarsening);
  lp_clustering.set_max_cluster_weight(max_cluster_weight);
  lp_clustering.set_desired_cluster_count(0);

  ENABLE_HEAP_PROFILER();
  START_HEAP_PROFILER("Label Propagation");
  TIMED_SCOPE("Label Propagation") {
    lp_clustering.compute_clustering(graph);
  };
  STOP_HEAP_PROFILER();
  DISABLE_HEAP_PROFILER();

  STOP_TIMER();

  cio::print_delimiter("Input Summary", '#');
  std::cout << "Execution mode:               " << ctx.parallel.num_threads << "\n";
  std::cout << "Seed:                         " << Random::get_seed() << "\n";
  cio::print_delimiter("Graph Compression", '-');
  print(ctx.compression, std::cout);
  cio::print_delimiter("Coarsening", '-');
  print(ctx.coarsening, std::cout);

  cio::print_delimiter("Result Summary");
  Timer::global().print_human_readable(std::cout);
  LOG;
  PRINT_HEAP_PROFILE(std::cout);

  return 0;
}
