//
// Created by badger on 5/8/24.
//

#include "networkit_utils.h"

namespace kaminpar::shm::sparsification::networkit_utils {
NetworKit::Graph toNetworKitGraph(const CSRGraph &g) {
  NetworKit::Graph nk_graph = NetworKit::Graph(g.n(), true, false);
  nk_graph.addNodes(g.n());
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      NodeID v = g.edge_target(e);
      if (u < v)
        nk_graph.addEdge(u, v, g.edge_weight(e));
    }
  }
  nk_graph.indexEdges();
  return nk_graph;
}
}
