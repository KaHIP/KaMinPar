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

#include "kaminpar-common/logger.h"

#include "apps/io/shm_compressed_graph_binary.h"
#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  // Parse CLI arguments
  std::string graph_filename;
  std::string compressed_graph_filename;
  int num_threads = 1;

  CLI::App app("Shared-memory graph compression tool");
  app.add_option("-G, --graph", graph_filename, "Input graph in METIS format")->required();
  app.add_option("--out", compressed_graph_filename, "Ouput file for saving the compressed graph")
      ->required();
  app.add_option("-t,--threads", num_threads, "Number of threads");
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  LOG << "Reading input graph...";
  CompressedGraph graph = io::metis::compress_read<false>(graph_filename);

  LOG << "Writing compressed graph...";
  io::compressed_binary::write(compressed_graph_filename, graph);

  return 0;
}
