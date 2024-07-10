/*******************************************************************************
 * Graph compression tool for the shared-memory algorithm.
 *
 * @file:   shm_graph_compression_tool.cc
 * @author: Daniel Salwasser
 * @date:   14.12.2023
 ******************************************************************************/
// clang-format off
#include <kaminpar-cli/kaminpar_arguments.h>
// clang-format on

#include <tbb/global_control.h>

#include "kaminpar-shm/context_io.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/logger.h"

#include "apps/io/metis_parser.h"
#include "apps/io/parhip_parser.h"
#include "apps/io/shm_compressed_graph_binary.h"
#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::io;

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  std::string graph_filename;
  std::string compressed_graph_filename;
  GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;
  NodeOrdering node_ordering = NodeOrdering::NATURAL;
  int num_threads = 1;

  CLI::App app("Shared-memory graph compression tool");
  app.add_option("-G,--graph", graph_filename, "Input graph in METIS format")->required();
  app.add_option("--out", compressed_graph_filename, "Ouput file for saving the compressed graph")
      ->required();
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)");
  app.add_option("--node-order", node_ordering)
      ->transform(CLI::CheckedTransformer(get_node_orderings()).description(""))
      ->description(R"(Criteria by which the nodes of the graph are sorted and rearranged:
  - natural:     keep node order of the graph (do not rearrange)
  - deg-buckets: sort nodes by degree bucket and rearrange accordingly
  - implicit-deg-buckets: nodes of the input graph are sorted by deg-buckets order)")
      ->capture_default_str();
  app.add_option("-t,--threads", num_threads, "Number of threads");
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  LOG << "Reading input graph...";

  CompressedGraph graph = [&] {
    switch (graph_file_format) {
    case GraphFileFormat::METIS:
      return *metis::compress_read(
          graph_filename, node_ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS, false
      );
    case GraphFileFormat::PARHIP:
      return parhip::compressed_read_parallel(graph_filename, node_ordering);
    default:
      __builtin_unreachable();
    }
  }();

  LOG << "Writing compressed graph...";
  io::compressed_binary::write(compressed_graph_filename, graph);

  return 0;
}
