/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "apps/apps.h"
#include "kaminpar/application/arguments.h"
#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/coarsening/parallel_label_propagation_coarsener.h"
#include "kaminpar/context.h"
#include "kaminpar/io.h"
#include "kaminpar/utility/timer.h"

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
  Timer::global().enable_fine();
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
      LabelPropagationClustering lp_core{graph.n(), ctx.coarsening.enforce_contraction_limit ? 0.5 : 0.0,
                                         ctx.coarsening.lp};
      const auto max_cluster_weight = compute_max_cluster_weight(graph, ctx.partition, ctx.coarsening);
      LOG << "max_cluster_weight=" << max_cluster_weight << " num_iterations=" << ctx.coarsening.lp.num_iterations;
      START_TIMER("Label Propagation");
      lp_core.cluster(graph, max_cluster_weight, ctx.coarsening.lp.num_iterations);
      STOP_TIMER();
    } else {
      ParallelLabelPropagationCoarsener coarsener{graph, ctx.coarsening};

      const Graph *c_graph = &graph;
      bool shrunk = true;

      while (shrunk) {
        const auto max_cluster_weight = compute_max_cluster_weight(*c_graph, ctx.partition, ctx.coarsening);
        const auto result = coarsener.coarsen(max_cluster_weight);
        if (ctx.coarsening.should_converge(c_graph->n(), result.first->n())) { break; }
        std::tie(c_graph, shrunk) = result;
      }
    }
  }

  Timer::global().print_machine_readable(std::cout);
  Timer::global().print_human_readable(std::cout);
}