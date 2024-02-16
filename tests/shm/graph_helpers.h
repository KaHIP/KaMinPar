#pragma once

#include <gmock/gmock.h>

#include "tests/shm/graph_builder.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/assert.h"
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
  return Graph{static_array::create_from(nodes), static_array::create_from(edges), {}, {}, sorted};
}

inline Graph create_graph(
    const std::vector<EdgeID> &nodes,
    const std::vector<NodeID> &edges,
    const std::vector<NodeWeight> &node_weights,
    const std::vector<EdgeWeight> &edge_weights,
    const bool sorted = false
) {
  return Graph{
      static_array::create_from(nodes),
      static_array::create_from(edges),
      static_array::create_from(node_weights),
      static_array::create_from(edge_weights),
      sorted};
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

inline Graph change_node_weight(Graph graph, const NodeID u, const NodeWeight new_node_weight) {
  auto node_weights = graph.take_raw_node_weights();
  node_weights[u] = new_node_weight;
  return Graph{
      graph.take_raw_nodes(),
      graph.take_raw_edges(),
      std::move(node_weights),
      graph.take_raw_edge_weights(),
      graph.sorted()};
}

inline Graph
change_edge_weight(Graph graph, const NodeID u, const NodeID v, const EdgeWeight new_edge_weight) {
  const EdgeID forward_edge = find_edge_by_endpoints(graph, u, v);
  const EdgeID backward_edge = find_edge_by_endpoints(graph, v, u);
  KASSERT(forward_edge != kInvalidEdgeID);
  KASSERT(backward_edge != kInvalidEdgeID);

  auto edge_weights = graph.take_raw_edge_weights();
  KASSERT(edge_weights[forward_edge] == edge_weights[backward_edge]);

  edge_weights[forward_edge] = new_edge_weight;
  edge_weights[backward_edge] = new_edge_weight;

  return Graph{
      graph.take_raw_nodes(),
      graph.take_raw_edges(),
      graph.take_raw_node_weights(),
      std::move(edge_weights),
      graph.sorted()};
}

inline Graph assign_exponential_weights(
    Graph graph, const bool assign_node_weights, const bool assign_edge_weights
) {
  KASSERT(
      !assign_node_weights || graph.n() <= std::numeric_limits<NodeWeight>::digits -
                                               std::numeric_limits<NodeWeight>::is_signed,
      "Cannot assign exponential node weights: graph has too many nodes",
      assert::always
  );
  KASSERT(
      !assign_edge_weights || graph.m() <= std::numeric_limits<EdgeWeight>::digits -
                                               std::numeric_limits<EdgeWeight>::is_signed,
      "Cannot assign exponential edge weights: graph has too many edges",
      assert::always
  );

  auto node_weights = graph.take_raw_node_weights();
  if (assign_node_weights) {
    for (const NodeID u : graph.nodes()) {
      node_weights[u] = 1 << u;
    }
  }

  auto edge_weights = graph.take_raw_edge_weights();
  if (assign_edge_weights) {
    for (const NodeID u : graph.nodes()) {
      for (const auto [e, v] : graph.neighbors(u)) {
        if (v > u) {
          continue;
        }
        edge_weights[e] = 1 << e;
        for (const auto [e_prime, u_prime] : graph.neighbors(v)) {
          if (u == u_prime) {
            edge_weights[e_prime] = edge_weights[e];
          }
        }
      }
    }
  }

  return Graph{
      graph.take_raw_nodes(),
      graph.take_raw_edges(),
      std::move(node_weights),
      std::move(edge_weights),
      graph.sorted()};
}
} // namespace kaminpar::shm::testing
