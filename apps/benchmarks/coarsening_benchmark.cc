/*******************************************************************************
 * @file:   coarsening_benchmark.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Performance benchmark for coarsening.
 ******************************************************************************/
#include "apps/apps.h"
#include "kaminpar/application/arguments.h"
#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/coarsening/cluster_coarsener.h"
#include "kaminpar/coarsening/label_propagation_clustering.h"
#include "kaminpar/context.h"
#include "kaminpar/io.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

using namespace kaminpar;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  ctx.partition.k = 8; // set some default for K, can be overwritten by CLI argument

  // parse command line arguments
  bool flat = false; // if true, only run label propagation on the input graph
  std::size_t num_iterations = 1;

  Arguments args;
  app::create_mandatory_context_options(ctx, args, "Mandatory");
  args.group("Benchmark", "b")
      .argument("flat", "If set, only run label propagation on the input graph.", &flat, 'f')
      .argument("iterations", "Number of benchmark iterations.", &num_iterations, 'i');
  app::create_parallel_context_options(ctx, args, "Parallel", "p");
  app::create_coarsening_context_options(ctx.coarsening, args, "Coarsening", "c");
  app::create_lp_coarsening_context_options(ctx.coarsening.lp, args, "Coarsening -> Label Propagation", "c-lp");
  args.parse(argc, argv);

  ALWAYS_ASSERT(!std::ifstream(ctx.graph_filename) == false)
      << "Graph file cannot be read. Ensure that the file exists and is readable.";

  // init components
  init_numa();
  auto gc = init_parallelism(ctx.parallel.num_threads);
  GLOBAL_TIMER.enable(TIMER_BENCHMARK);
  Randomize::seed = ctx.seed;

  // load graph
  START_TIMER("IO");
  Graph graph = io::metis::read(ctx.graph_filename);
  ctx.setup(graph);
  STOP_TIMER();
  LOG << "GRAPH n=" << graph.n() << " m=" << graph.m();

  for (std::size_t i = 0; i < num_iterations; ++i) {
    LOG << "Iteration " << i;

    if (flat) {
      const auto max_cluster_weight = compute_max_cluster_weight(graph, ctx.partition, ctx.coarsening);
      LOG << "max_cluster_weight=" << max_cluster_weight << " num_iterations=" << ctx.coarsening.lp.num_iterations;
      START_TIMER("Label Propagation");

      LabelPropagationClusteringAlgorithm lp_core{graph.n(), ctx.coarsening};
      lp_core.set_max_cluster_weight(max_cluster_weight);
      lp_core.compute_clustering(graph);
      STOP_TIMER();
    } else {
      ClusteringCoarsener coarsener{std::make_unique<LabelPropagationClusteringAlgorithm>(graph.n(), ctx.coarsening),
                                    graph, ctx.coarsening};

      const Graph *c_graph = &graph;
      bool shrunk = true;

      while (shrunk) {
        const auto max_cluster_weight = compute_max_cluster_weight(*c_graph, ctx.partition, ctx.coarsening);
        const NodeID old_n = c_graph->n();
        const auto result = coarsener.compute_coarse_graph(max_cluster_weight, 0); // might invalidate c_graph ptr
        if (ctx.coarsening.coarsening_should_converge(old_n, result.first->n())) {
          break;
        }
        std::tie(c_graph, shrunk) = result;
      }
    }
  }

  Timer::global().print_machine_readable(std::cout);
  Timer::global().print_human_readable(std::cout);
}