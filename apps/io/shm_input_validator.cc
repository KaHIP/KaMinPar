/*******************************************************************************
 * Validator for undirected input graphs.
 *
 * @file:   shm_input_validator.cc
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#include "apps/io/shm_input_validator.h"

#include <algorithm>
#include <iostream>
#include <tuple>

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

template <typename ForwardIterator, typename T>
ForwardIterator binary_find(ForwardIterator begin, ForwardIterator end, const T &val) {
  const auto i = std::lower_bound(begin, end, val);
  return (i != end && (*i <= val)) ? i : end;
}

void validate_undirected_graph(
    const StaticArray<EdgeID> &nodes,
    const StaticArray<NodeID> &edges,
    const StaticArray<NodeWeight> &,
    const StaticArray<EdgeWeight> &edge_weights
) {
  const NodeID n = nodes.size() - 1;
  const EdgeID m = edges.size();

  // Create a copy of edges and edge weights for sorting
  StaticArray<std::tuple<NodeID, EdgeWeight>> edges_with_weights(m);
  tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
    const EdgeWeight weight = edge_weights.empty() ? 1 : edge_weights[e];
    edges_with_weights[e] = std::make_tuple(edges[e], weight);
  });

  // Sort outgoing edges of each node
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
    std::sort(
        edges_with_weights.begin() + nodes[u],
        edges_with_weights.begin() + nodes[u + 1],
        [&](const auto &lhs, const auto &rhs) { return std::get<0>(lhs) < std::get<0>(rhs); }
    );

    // Check for multi edges
    for (EdgeID e = nodes[u] + 1; e < nodes[u + 1]; ++e) {
      if (edges[e - 1] == edges[e]) {
        std::cerr << "node " << u + 1 << " has multiple edges to neighbor " << edges[e] << "\n";
        std::exit(1);
      }
    }
  });

  // Check for reverse edges
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
    for (EdgeID e = nodes[u]; e < nodes[u + 1]; ++e) {
      const auto [v, weight] = edges_with_weights[e];

      const auto it_begin = edges_with_weights.begin() + nodes[v];
      const auto it_end = edges_with_weights.begin() + nodes[v + 1];
      const auto rev = binary_find(it_begin, it_end, std::make_tuple(u, weight));

      if (rev == it_end) {
        std::cerr << "missing reverse edge: of edge " << u + 1 << " --> " << v + 1
                  << " (the reverse edge might exist but with an inconsistent "
                     "weight)\n";
        std::exit(1);
      }
    }
  });
}

} // namespace

void validate_undirected_graph(const Graph &graph) {
  SCOPED_HEAP_PROFILER("Validate Input Graph");
  SCOPED_TIMER("Validate input graph");

  if (auto *csr_graph = dynamic_cast<const CSRGraph *>(graph.underlying_graph());
      csr_graph != nullptr) {
    validate_undirected_graph(
        csr_graph->raw_nodes(),
        csr_graph->raw_edges(),
        csr_graph->raw_node_weights(),
        csr_graph->raw_edge_weights()
    );
  }
}

} // namespace kaminpar::shm
