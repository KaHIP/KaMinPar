/*******************************************************************************
 * Graph properties tool for the shared-memory algorithm.
 *
 * @file:   shm_graph_properties_tool.cc
 * @author: Daniel Salwasser
 * @date:   26.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/context_io.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/strutils.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

float average_degree(const Graph &graph) {
  std::size_t average_degree = 0;

  for (const NodeID node : graph.nodes()) {
    average_degree += graph.degree(node);
  }

  return average_degree / (float)graph.n();
}

NodeID isolated_nodes(const Graph &graph) {
  NodeID count = 0;

  for (const NodeID node : graph.nodes()) {
    if (graph.degree(node) == 0) {
      count++;
    }
  }

  return count;
}

void print_graph_properties(const Graph &graph, const Context ctx, std::ostream &out) {
  const float avg_deg = average_degree(graph);
  const NodeID isolated_node_count = isolated_nodes(graph);
  const std::size_t width = std::ceil(std::log10(
      std::max<std::size_t>({graph.n(), graph.m(), graph.max_degree(), isolated_node_count})
  ));

  cio::print_delimiter("Graph Properties", '#');
  out << "Graph:                        " << ctx.debug.graph_name << "\n";
  out << "  Number of nodes:            " << std::setw(width) << graph.n();
  if (graph.node_weighted()) {
    out << " (total weight: " << graph.total_node_weight() << ")\n";
  } else {
    out << " (unweighted)\n";
  }
  out << "  Number of edges:            " << std::setw(width) << graph.m();
  if (graph.edge_weighted()) {
    out << " (total weight: " << graph.total_edge_weight() << ")\n";
  } else {
    out << " (unweighted)\n";
  }
  out << "  Max degree:                 " << std::setw(width) << graph.max_degree() << '\n';
  out << "  Average degree:             " << std::setw(width) << avg_deg << '\n';
  out << "  Isolated nodes:             " << std::setw(width) << isolated_node_count << '\n';

  cio::print_delimiter("Graph Compression", '-');
  print(ctx.compression, out);
}

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  std::string graph_filename;

  CLI::App app("Shared-memory graph properties tool");
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS format")->required();
  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);

  Graph graph = io::read(graph_filename, ctx.compression.enabled, false, false);

  ctx.debug.graph_name = str::extract_basename(graph_filename);
  ctx.compression.setup(graph);

  print_graph_properties(graph, ctx, std::cout);

  return 0;
}
