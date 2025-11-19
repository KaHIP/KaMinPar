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

#include "kaminpar-io/kaminpar_io.h"

#include "kaminpar-shm/graphutils/permutator.h"

#include "kaminpar-common/logger.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::io;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  ctx.partition.k = 0;

  // Parse CLI arguments
  CLI::App app("Shared-memory graph rearrangement tool");

  std::string graph_filename;
  GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS format")->required();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(
          std::unordered_map<std::string, io::GraphFileFormat>{
              {"metis", io::GraphFileFormat::METIS},
              {"parhip", io::GraphFileFormat::PARHIP},
              {"compressed", io::GraphFileFormat::COMPRESSED},
          },
          CLI::ignore_case
      ))
      ->description(R"(Graph file format of the input graph:
  - metis
  - parhip)");

  std::string out_graph_filename;
  GraphFileFormat out_graph_file_format = io::GraphFileFormat::METIS;
  app.add_option("-O,--out", out_graph_filename, "Ouput file for saving the rearranged graph")
      ->required();
  app.add_option("--out-f,--out-graph-file-format", out_graph_file_format)
      ->transform(CLI::CheckedTransformer(
          std::unordered_map<std::string, io::GraphFileFormat>{
              {"metis", io::GraphFileFormat::METIS},
              {"parhip", io::GraphFileFormat::PARHIP},
          },
          CLI::ignore_case
      ))
      ->description(R"(Graph file format used for storing the rearranged graph:
  - metis
  - parhip)");

  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  create_partitioning_rearrangement_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);

  LOG << "Reading input graph...";
  auto input_graph = io::read_graph(graph_filename, graph_file_format, false, ctx.node_ordering);
  if (!input_graph) {
    LOG_ERROR << "Failed to read the input graph";
    return EXIT_FAILURE;
  }

  Graph graph(std::make_unique<CSRGraph>(std::move(*input_graph)));

  LOG << "Rearranging graph...";
  if (ctx.node_ordering == NodeOrdering::DEGREE_BUCKETS) {
    graph = graph::rearrange_by_degree_buckets(graph.csr_graph());
  }

  if (ctx.edge_ordering == EdgeOrdering::COMPRESSION) {
    graph::reorder_edges_by_compression(graph.csr_graph());
  }

  LOG << "Writing rearanged graph...";
  io::write_graph(out_graph_filename, out_graph_file_format, graph);

  return EXIT_SUCCESS;
}
