#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/io.h"
#include "kaminpar/utility/metrics.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <ranges>

using namespace kaminpar;

int main(int argc, char *argv[]) {
  std::string graph_filename{};
  std::string partition_filename{};
  bool print_block_weights{false};

  Arguments arguments;
  arguments.positional()
      .argument("graph", "Filename of graph in METIS format", &graph_filename)
      .argument("partition", "Filename of the partition to verify", &partition_filename);
  arguments.group("Optional options are").argument("block-weights", "Print block weights.", &print_block_weights);
  arguments.parse(argc, argv);

  Graph graph = io::metis::read(graph_filename);
  auto partition = io::partition::read(partition_filename);
  if (partition.size() != graph.n()) {
    FATAL_ERROR << "Graph has " << graph.n() << " nodes, but partition has " << partition.size() << " elements";
  }

  // copy to StaticArray data structure
  StaticArray<BlockID> static_partition(partition.size());
  std::ranges::copy(partition, static_partition.begin());

  const BlockID k = *std::max_element(partition.begin(), partition.end()) + 1;
  const PartitionedGraph p_graph(graph, k, std::move(static_partition));

  if (print_block_weights) { LOG << logger::TABLE << p_graph.block_weights(); }

  LOG << "RESULT k=" << k << " cut=" << metrics::edge_cut(p_graph) << " imbalance=" << metrics::imbalance(p_graph);
  return EXIT_SUCCESS;
}
