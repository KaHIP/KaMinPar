/*******************************************************************************
 * @file:   balancing_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Benchmark for the shared-memory balancing algorithm.
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-format on

#include "kaminpar/arguments.h"
#include "kaminpar/coarsening/cluster_coarsener.h"
#include "kaminpar/coarsening/lp_clustering.h"
#include "kaminpar/context.h"
#include "kaminpar/factories.h"
#include "kaminpar/io.h"
#include "kaminpar/metrics.h"
#include "kaminpar/presets.h"

#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  std::string partition_filename;

  CLI::App app;
  app.add_option("-G", ctx.graph_filename);
  app.add_option("-P", partition_filename);
  app.add_option("-t", ctx.parallel.num_threads);
  app.add_option("-e", ctx.partition.epsilon);
  create_balancer_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  if (!std::ifstream(ctx.graph_filename)) {
    FATAL_ERROR << "Graph file cannot be read. Ensure that the file exists and "
                   "is readable.";
  }
  if (!std::ifstream(partition_filename)) {
    FATAL_ERROR << "Partition file cannot be read. Ensure that the file exists "
                   "and is readable.";
  }

  init_numa();
  auto gc = init_parallelism(ctx.parallel.num_threads);
  Random::seed = ctx.seed;

  START_TIMER("IO");
  Graph graph = shm::io::metis::read<true>(ctx.graph_filename);
  auto partition = shm::io::partition::read<StaticArray<BlockID>>(partition_filename);
  const BlockID k = *std::max_element(partition.begin(), partition.end()) + 1;
  KASSERT(partition.size() == graph.n(), "bad partition size", assert::always);
  PartitionedGraph p_graph(graph, k, std::move(partition));
  STOP_TIMER();

  const EdgeWeight cut_before = metrics::edge_cut(p_graph);
  const double imbalance_before = metrics::imbalance(p_graph);

  LOG << "Read partitioned graph with: "
      << "n=" << graph.n() << " "
      << "m=" << graph.m() << " "
      << "k=" << k << " "
      << "cut=" << cut_before << " "
      << "imbalance=" << imbalance_before;

  ctx.partition.k = k;
  ctx.setup(graph);

  auto balancer = factory::create_balancer(graph, ctx.partition, ctx.refinement);

  TIMED_SCOPE("Balancer") {
    TIMED_SCOPE("Initialization") {
      balancer->initialize(p_graph);
    };
    TIMED_SCOPE("Balancing") {
      balancer->balance(p_graph, ctx.partition);
    };
  };

  const EdgeWeight cut_after = metrics::edge_cut(p_graph);
  const double imbalance_after = metrics::imbalance(p_graph);

  LOG << "Result: "
      << "cut=" << cut_after << " "
      << "imbalance=" << imbalance_after;

  Timer::global().print_machine_readable(std::cout);
  Timer::global().print_human_readable(std::cout);
}
