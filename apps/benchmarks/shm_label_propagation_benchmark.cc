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
#include "kaminpar-shm/partition_utils.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  // Create context
  Context ctx = create_default_context();

  // Parse CLI arguments
  std::string graph_filename;
  std::string partition_filename;
  int num_threads = 1;

  CLI::App app("Shared-memory LP benchmark");
  app.add_option("-G,--graph", graph_filename, "Graph file")->required();
  app.add_option("-t,--threads", num_threads, "Number of threads");
  create_lp_coarsening_options(&app, ctx);
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  Graph graph = io::read(graph_filename, ctx.compression.enabled, false);
  LPClustering lp_clustering(graph.n(), ctx.coarsening);

  const NodeWeight max_cluster_weight =
      compute_max_cluster_weight(ctx.coarsening, graph, ctx.partition);
  lp_clustering.set_max_cluster_weight(max_cluster_weight);
  lp_clustering.set_desired_cluster_count(0);

  ENABLE_HEAP_PROFILER();
  START_HEAP_PROFILER("Label Propagation");
  TIMED_SCOPE("Label Propagation") {
    lp_clustering.compute_clustering(graph);
  };
  STOP_HEAP_PROFILER();
  DISABLE_HEAP_PROFILER();

  STOP_TIMER(); // Stop root timer
  Timer::global().print_human_readable(std::cout);
  LOG;
  PRINT_HEAP_PROFILE(std::cout);

  return 0;
}
