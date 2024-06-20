
#include "sparsification_utils.h"
namespace kaminpar::shm::sparsification::utils {

StaticArray<EdgeID> sort_by_traget(const CSRGraph &g) {
  auto permutation = StaticArray<EdgeID>(g.m());
  for (auto e : g.edges())
    permutation[e] = e;
  for (NodeID u : g.nodes()){
    std::sort(
        permutation.begin() + g.raw_nodes()[u],
        permutation.begin() + g.raw_nodes()[u + 1],
        [&](EdgeID e1, EdgeID e2) { return g.edge_target(e1) <= g.edge_target(e2); }
    );
  }
  return permutation;
}

void for_edges_with_endpoints(
    const CSRGraph &g, std::function<void(EdgeID, NodeID, NodeID)> function
) {
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      NodeID v = g.edge_target(e);
      function(e, u, v);
    }
  }
}
void for_upward_edges(const CSRGraph &g, std::function<void(EdgeID)> function) {
  for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (u < v)
      function(e);
  });
}

} // namespace kaminpar::shm::sparsification::utils
