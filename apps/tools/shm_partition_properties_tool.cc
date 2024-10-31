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

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/strutils.h"

#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

int main(int argc, char *argv[]) {
  Context ctx = create_default_context();
  std::string graph_filename;
  std::string partition_filename;
  std::string block_sizes_filename;

  io::GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;

  CLI::App app("Shared-memory partition properties tool");
  app.add_option("-G,--graph", graph_filename, "Input graph")->required();

  auto *partition_group = app.group("Partition options:"); //->require_option(1);
  partition_group->add_option(
      "-P,--partition", partition_filename, "Partition (block of one node per line)"
  );
  partition_group->add_option(
      "--block-sizes", block_sizes_filename, "Block sizes (one size per line)"
  );

  app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads");
  app.add_option("-f,--graph-file-format", graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)");
  create_graph_compression_options(&app, ctx);
  CLI11_PARSE(app, argc, argv);

  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads);

  Graph graph =
      io::read(graph_filename, graph_file_format, NodeOrdering::NATURAL, ctx.compression.enabled);

  ctx.debug.graph_name = str::extract_basename(graph_filename);
  ctx.compression.setup(graph);

  LOG << "Graph:            " << ctx.debug.graph_name;

  StaticArray<BlockID> partition;
  if (!partition_filename.empty()) {
    LOG << "Partition:        " << str::extract_basename(partition_filename);

    partition = io::partition::read(partition_filename);
  } else if (!block_sizes_filename.empty()) {
    LOG << "Block sizes:      " << str::extract_basename(block_sizes_filename);

    partition = io::partition::read_block_sizes(block_sizes_filename);
  } else {
    LOG_ERROR << "No partition or block sizes provided";
    return 1;
  }

  const BlockID k = *std::max_element(partition.begin(), partition.end()) + 1;
  PartitionedGraph p_graph(graph, k, std::move(partition));

  LOG << "Number of blocks: " << k;
  LOG << "Edge cut:         " << metrics::edge_cut(p_graph);
  LOG << "Imbalance:        " << metrics::imbalance(p_graph);

  return 0;
}
