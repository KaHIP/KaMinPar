#pragma once

#include <functional>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification::utils {

StaticArray<EdgeID> sort_by_traget(const CSRGraph &g);

void for_edges_with_endpoints(
    const CSRGraph &g, std::function<void(EdgeID, NodeID, NodeID)> function
);
void for_upward_edges(const CSRGraph &g, std::function<void(EdgeID)>);
void for_downward_edges(const CSRGraph &g, std::function<void(EdgeID)>);

template <typename Lambda>
inline void p_for_edges_with_endpoints(const CSRGraph &g, Lambda function) {
  g.pfor_nodes([&](NodeID u) {
    g.pfor_neighbors(u, g.n() - 1, 123, [&](EdgeID e, NodeID v, EdgeWeight e_weight) {
      function(e, u, v);
    });
  });
}
template <typename Lambda> inline void p_for_upward_edges(const CSRGraph &g, Lambda function) {
  p_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (u < v)
      function(e);
  });
}
template <typename Lambda> inline void p_for_downward_edges(const CSRGraph &g, Lambda function) {
  p_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (u > v)
      function(e);
  });
}
} // namespace kaminpar::shm::sparsification::utils
