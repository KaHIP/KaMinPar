#include <string>

#include <kaminpar_networkit.h>
#include <networkit/community/EdgeCut.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/MetisGraphReader.hpp>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "This example uses NetworKit to load a graph, then partition it using KaMinPar.\n";
    std::cerr << "Usage: ./ShmNetworKitExample <graph.metis> <k>\n";
    return 1;
  }

  const std::string graph_filename = argv[1];
  const kaminpar::shm::BlockID k = std::stoi(argv[2]);

  NetworKit::Graph graph = NetworKit::METISGraphReader().read(graph_filename);

  kaminpar::KaMinParNetworKit kaminpar(graph);
  NetworKit::Partition partition = kaminpar.computePartition(k);

  const double cut = NetworKit::EdgeCut().getQuality(partition, graph);
  std::cout << "Edge cut via NetworKit::EdgeCut::getQuality(): " << cut << std::endl;

  NetworKit::edgeweight manual_cut = 0;
  graph.forEdges([&](NetworKit::node u, NetworKit::node v, NetworKit::edgeweight weight) {
    if (!partition.inSameSubset(u, v)) {
      manual_cut += weight;
    }
  });
  std::cout << "Edge cut via manual computation: " << manual_cut << std::endl;
}
