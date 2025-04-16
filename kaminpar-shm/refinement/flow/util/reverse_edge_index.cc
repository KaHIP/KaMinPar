#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

[[nodiscard]] StaticArray<EdgeID> compute_reverse_edge_index(const CSRGraph &graph) {
  const NodeID num_nodes = graph.n();
  const NodeID num_edges = graph.m();

  StaticArray<NodeID> index(num_edges, static_array::noinit);

  struct EdgeHasher {
    [[nodiscard]] std::size_t operator()(const std::pair<NodeID, NodeID> &edge) const noexcept {
      return edge.first ^ (edge.second << 1);
    }
  };
  std::unordered_map<std::pair<NodeID, NodeID>, EdgeID, EdgeHasher> edge_table;

  for (NodeID u = 0; u < num_nodes; u++) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (u < v) {
        edge_table.insert({{u, v}, e});
      } else {
        auto it = edge_table.find({v, u});
        KASSERT(it != edge_table.end());

        const EdgeID e_reverse = it->second;
        index[e] = e_reverse;
        index[e_reverse] = e;

        edge_table.erase(it);
      }
    });
  }

  return index;
}

} // namespace kaminpar::shm
