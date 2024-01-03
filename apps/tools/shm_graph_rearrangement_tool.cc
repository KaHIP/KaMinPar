/*******************************************************************************
 * CSR graph rearrangement tool for the shared-memory algorithm.
 *
 * @file:   shm_graph_rearrangement_tool.cc
 * @author: Daniel Salwasser
 * @date:   14.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/graphutils/permutator.h"

#include "kaminpar-common/logger.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  ctx.partition.k = 0;

  // Parse CLI arguments
  std::string graph_filename;
  std::string out_graph_filename;

  CLI::App app("Shared-memory graph rearrangement tool");
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS format")->required();
  app.add_option("-O,--out", out_graph_filename, "Ouput file for saving the rearranged graph")
      ->required();
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  create_partitioning_rearrangement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);

  LOG << "Reading input graph...";
  CSRGraph csr_graph = io::metis::csr_read<false>(graph_filename);
  Graph graph(std::make_unique<CSRGraph>(std::move(csr_graph)));

  LOG << "Rearranging graph...";
  if (ctx.node_ordering == NodeOrdering::DEGREE_BUCKETS) {
    graph = graph::rearrange_by_degree_buckets(ctx, std::move(graph));
    graph::integrate_isolated_nodes(graph, ctx.partition.epsilon, ctx);
  }

  if (ctx.edge_ordering == EdgeOrdering::COMPRESSION) {
    graph::reorder_edges_by_compression(*dynamic_cast<CSRGraph *>(graph.underlying_graph()));
  }

  LOG << "Writing graph...";
  io::metis::write(out_graph_filename, graph);

  return 0;
}
