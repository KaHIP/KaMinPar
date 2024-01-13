#pragma once

#include <gmock/gmock.h>

#include "tests/shm/graph_builder.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assertion_levels.h"
#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm::testing {
namespace graphs {
inline PartitionedGraph p_graph(const Graph &graph, const BlockID k) {
  return PartitionedGraph(graph, k, static_array::create_from(std::vector<BlockID>(graph.n())));
}
} // namespace graphs

//
// Convenience functions to create Graph / PartitionedGraph from initializer
// lists
//

inline Graph create_graph(
    const std::vector<EdgeID> &nodes, const std::vector<NodeID> &edges, const bool sorted = false
) {
  return Graph(std::make_unique<CSRGraph>(
      static_array::create_from(nodes),
      static_array::create_from(edges),
      StaticArray<NodeWeight>(),
      StaticArray<EdgeWeight>(),
      sorted
  ));
}

inline Graph create_graph(
    const std::vector<EdgeID> &nodes,
    const std::vector<NodeID> &edges,
    const std::vector<NodeWeight> &node_weights,
    const std::vector<EdgeWeight> &edge_weights,
    const bool sorted = false
) {
  return Graph(std::make_unique<CSRGraph>(
      static_array::create_from(nodes),
      static_array::create_from(edges),
      static_array::create_from(node_weights),
      static_array::create_from(edge_weights),
      sorted
  ));
}

inline PartitionedGraph
create_p_graph(const Graph &graph, const BlockID k, const std::vector<BlockID> &partition) {
  return PartitionedGraph{graph, k, static_array::create_from(partition)};
}

inline PartitionedGraph
create_p_graph(const Graph *graph, const BlockID k, const std::vector<BlockID> &partition) {
  return create_p_graph(*graph, k, partition);
}

inline PartitionedGraph create_p_graph(const Graph &graph, const BlockID k) {
  return PartitionedGraph{graph, k};
}

inline PartitionedGraph create_p_graph(const Graph *graph, const BlockID k) {
  return create_p_graph(*graph, k);
}

template <typename T> StaticArray<T> create_static_array(const std::vector<T> &elements) {
  StaticArray<T> arr(elements.size());
  for (std::size_t i = 0; i < elements.size(); ++i) {
    arr[i] = elements[i];
  }
  return arr;
}

inline EdgeID find_edge_by_endpoints(const Graph &graph, const NodeID u, const NodeID v) {
  for (const auto [e, v_prime] : graph.neighbors(u)) {
    if (v == v_prime) {
      return e;
    }
  }
  return kInvalidEdgeID;
}

inline std::vector<NodeID> degrees(const Graph &graph) {
  std::vector<NodeID> degrees(graph.n());
  for (const NodeID u : graph.nodes()) {
    degrees[u] = graph.degree(u);
  }
  return degrees;
}

inline void change_node_weight(Graph &graph, const NodeID u, const NodeWeight new_node_weight) {
  auto &node_weights = graph.raw_node_weights();
  node_weights[u] = new_node_weight;
}

} // namespace kaminpar::shm::testing
