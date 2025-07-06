#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <kaminpar.h>
#include <kaminpar_io.h>

int main(int argc, char **argv) {
  using namespace kaminpar;
  using namespace kaminpar::shm;

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <num-blocks> <input-file> <output-file>" << std::endl;
    return EXIT_FAILURE;
  }

  const int num_blocks = std::stoi(argv[1]);
  const char *filename = argv[2];
  const char *output = argv[3];

  std::optional<Graph> graph = io::read_graph(filename, io::GraphFileFormat::METIS);
  if (!graph.has_value()) {
    std::cerr << "Failed to read graph from file " << filename << std::endl;
    return EXIT_FAILURE;
  }

  const NodeID num_nodes = graph->n();
  std::vector<BlockID> partition(num_nodes);

  KaMinPar kaminpar;
  kaminpar.set_graph(std::move(graph.value()));
  kaminpar.set_k(num_blocks);
  kaminpar.set_uniform_max_block_weights(0.03);

  kaminpar.compute_partition(partition);

  io::write_partition(output, partition);

  return EXIT_SUCCESS;
}
