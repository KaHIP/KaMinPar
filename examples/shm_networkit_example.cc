#include <string>

#include <kaminpar_networkit.h>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/MetisGraphReader.hpp>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./ShmNetworKitExample <graph.metis> <k>\n";
    return 1;
  }

  const std::string graph_filename = argv[1];
  const kaminpar::shm::BlockID k = std::stoi(argv[2]);

  NetworKit::Graph graph = NetworKit::METISGraphReader().read(graph_filename);
  std::vector<kaminpar::shm::BlockID> partition(graph.numberOfNodes());

  kaminpar::KaMinParNetworKit kaminpar(4, kaminpar::shm::create_default_context());
  kaminpar.set_output_level(kaminpar::OutputLevel::DEBUG);
  kaminpar.copy_graph(graph);
  const kaminpar::shm::EdgeWeight cut = kaminpar.compute_partition(k, 0.03, partition);

  std::cout << std::endl;
  std::cout << "Edge cut: " << cut << std::endl;
}
