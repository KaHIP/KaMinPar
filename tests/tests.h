/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "graph_builder.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"

#include "gmock/gmock.h"

using ::testing::Test;

namespace kaminpar::test {
//
// Convenience functions to create Graph / PartitionedGraph from initializer lists
//

Graph create_graph(const std::vector<EdgeID> &nodes, const std::vector<NodeID> &edges, const bool sorted = false) {
  return Graph{from_vec(nodes), from_vec(edges), {}, {}, sorted};
}

Graph create_graph(const std::vector<EdgeID> &nodes, const std::vector<NodeID> &edges,
                   const std::vector<NodeWeight> &node_weights, const std::vector<EdgeWeight> &edge_weights,
                   const bool sorted = false) {
  return Graph{from_vec(nodes), from_vec(edges), from_vec(node_weights), from_vec(edge_weights), sorted};
}

PartitionedGraph create_p_graph(const Graph &graph, const BlockID k, const std::vector<BlockID> &partition) {
  return PartitionedGraph{graph, k, from_vec(partition)};
}

PartitionedGraph create_p_graph(const Graph &graph, const BlockID k, const std::vector<BlockID> &partition,
                                scalable_vector<BlockID> final_ks) {
  return PartitionedGraph{graph, k, from_vec(partition), std::move(final_ks)};
}

PartitionedGraph create_p_graph(const Graph *graph, const BlockID k, const std::vector<BlockID> &partition) {
  return create_p_graph(*graph, k, partition);
}

PartitionedGraph create_p_graph(const Graph &graph, const BlockID k) { return PartitionedGraph{graph, k}; }

PartitionedGraph create_p_graph(const Graph *graph, const BlockID k) { return create_p_graph(*graph, k); }

template<typename T>
StaticArray<T> create_static_array(const std::vector<T> &elements) {
  StaticArray<T> arr(elements.size());
  for (std::size_t i = 0; i < elements.size(); ++i) { arr[i] = elements[i]; }
  return arr;
}

EdgeID find_edge_by_endpoints(const Graph &graph, const NodeID u, const NodeID v) {
  for (const auto [e, v_prime] : graph.neighbors(u)) {
    if (v == v_prime) { return e; }
  }
  return kInvalidEdgeID;
}

std::vector<Degree> degrees(const Graph &graph) {
  std::vector<Degree> degrees(graph.n());
  for (const NodeID u : graph.nodes()) { degrees[u] = graph.degree(u); }
  return degrees;
}

Graph change_node_weight(Graph graph, const NodeID u, const NodeWeight new_node_weight) {
  auto node_weights = graph.take_raw_node_weights();
  node_weights[u] = new_node_weight;
  return Graph{graph.take_raw_nodes(), graph.take_raw_edges(), std::move(node_weights), graph.take_raw_edge_weights(),
               graph.sorted()};
}

Graph change_edge_weight(Graph graph, const NodeID u, const NodeID v, const EdgeWeight new_edge_weight) {
  const EdgeID forward_edge = find_edge_by_endpoints(graph, u, v);
  const EdgeID backward_edge = find_edge_by_endpoints(graph, v, u);
  ASSERT(forward_edge != kInvalidEdgeID);
  ASSERT(backward_edge != kInvalidEdgeID);

  auto edge_weights = graph.take_raw_edge_weights();
  ASSERT(edge_weights[forward_edge] == edge_weights[backward_edge]);

  edge_weights[forward_edge] = new_edge_weight;
  edge_weights[backward_edge] = new_edge_weight;

  return Graph{graph.take_raw_nodes(), graph.take_raw_edges(), graph.take_raw_node_weights(), std::move(edge_weights),
               graph.sorted()};
}

Graph assign_exponential_weights(Graph graph, const bool assign_node_weights, const bool assign_edge_weights) {
  ALWAYS_ASSERT(!assign_node_weights ||
                graph.n() <= std::numeric_limits<NodeWeight>::digits - std::numeric_limits<NodeWeight>::is_signed)
      << "Cannot assign exponential node weights: graph has too many nodes";
  ALWAYS_ASSERT(!assign_edge_weights ||
                graph.m() <= std::numeric_limits<EdgeWeight>::digits - std::numeric_limits<EdgeWeight>::is_signed)
      << "Cannot assign exponential edge weights: graph has too many edges";

  auto node_weights = graph.take_raw_node_weights();
  if (assign_node_weights) {
    for (const NodeID u : graph.nodes()) { node_weights[u] = 1 << u; }
  }

  auto edge_weights = graph.take_raw_edge_weights();
  if (assign_edge_weights) {
    for (const NodeID u : graph.nodes()) {
      for (const auto [e, v] : graph.neighbors(u)) {
        if (v > u) { continue; }
        edge_weights[e] = 1 << e;
        for (const auto [e_prime, u_prime] : graph.neighbors(v)) {
          if (u == u_prime) { edge_weights[e_prime] = edge_weights[e]; }
        }
      }
    }
  }

  return Graph{graph.take_raw_nodes(), graph.take_raw_edges(), std::move(node_weights), std::move(edge_weights),
               graph.sorted()};
}

std::string test_instance(const std::string &name) {
  using namespace std::literals;
  return "test_instances/"s + name;
}

template<typename View>
auto view_to_vector(const View &&view) {
  std::vector<std::decay_t<decltype(*view.begin())>> vec;
  for (const auto &e : view) { vec.push_back(e); }
  return vec;
}

Context create_context(const Graph &graph, const BlockID k = 2, const double epsilon = 0.03) {
  Context context;
  context.partition.k = k;
  context.partition.epsilon = epsilon;
  context.setup(graph);
  return context;
}

/*!
 * Builds a single graph that contains all graphs as induced subgraphs.
 *
 * @param graphs A list of graphs that should be copied.
 * @param connect_graphs If true, the first node of each graph is connected to a clique with edges of weight 1.
 *  Otherwise, the induced subgraphs are disconnected.
 * @return A single graph containing all other graphs.
 */
Graph merge_graphs(std::initializer_list<Graph *> graphs, const bool connect_graphs = false) {
  const NodeID n = std::accumulate(graphs.begin(), graphs.end(), 0,
                                   [&](const NodeID acc, const Graph *graph) { return acc + graph->n(); });
  const EdgeID m = std::accumulate(graphs.begin(), graphs.end(), 0,
                                   [&](const EdgeID acc, const Graph *graph) { return acc + graph->m(); });
  GraphBuilder builder(n, m);

  NodeID offset = 0;
  for (const Graph *graph : graphs) {
    ASSERT(graph->n() > 0);
    builder.new_node(graph->node_weight(0));

    if (connect_graphs) {
      NodeID first_node_in_other_graph = 0;
      for (const Graph *other_graph : graphs) {
        if (other_graph == graph) { continue; }
        ASSERT(other_graph->n() > 0);
        builder.new_edge(first_node_in_other_graph);
        first_node_in_other_graph += other_graph->n();
      }
    }

    for (const NodeID u : graph->nodes()) {
      if (u > 0) { builder.new_node(graph->node_weight(u)); }
      for (const auto [e, v] : graph->neighbors(u)) { builder.new_edge(offset + v, graph->edge_weight(e)); }
    }

    offset += graph->n();
  }

  return builder.build();
}

namespace graphs {
/*!
 * Builds a graph with `n` nodes and zero edges.
 *
 * @param n Number of nodes in the graph.
 * @return Graph on `n` nodes and zero edges.
 */
template<typename... GraphArgs>
Graph empty(const NodeID n, GraphArgs &&...graph_args) {
  GraphBuilder builder(n, 0);
  for (NodeID u = 0; u < n; ++u) { builder.new_node(); }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds a 2D grid graph.
 *
 * Edge nodes have degree 2, border nodes have degree 3, inner nodes have degree 4. The 2D grid graph with parameters
 * `(u = 3, v = 3)` looks like the following, where digits represent nodes:
 *
 * ```
 * 0--1--2
 * |  |  |
 * 3--4--5
 * |  |  |
 * 6--7--8
 * ```
 *
 * @param u Number of rows.
 * @param v Number of columns.
 * @return Grid graph on `u * v` nodes.
 */
template<typename... GraphArgs>
Graph grid(const NodeID u, const NodeID v, GraphArgs &&...graph_args) { // u x v grid
  GraphBuilder builder;
  for (NodeID i = 0; i < u; ++i) {
    const bool first_row = (i == 0);
    const bool last_row = (i + 1 == u);
    for (NodeID j = 0; j < v; ++j) {
      const bool first_column = (j == 0);
      const bool last_column = (j + 1 == v);
      const NodeID node = builder.new_node();
      if (!first_row) { builder.new_edge(node - v); }
      if (!last_row) { builder.new_edge(node + v); }
      if (!first_column) { builder.new_edge(node - 1); }
      if (!last_column) { builder.new_edge(node + 1); }
    }
  }

  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds a path on `length` nodes, i.e., a `1 * length` 2D grid.
 *
 * @param length Length of the path.
 * @return Path on `length` nodes.
 */
template<typename... GraphArgs>
Graph path(const NodeID length, GraphArgs &&...graph_args) {
  return grid(length, 1, std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds the complete bipartite graph on `n + m` nodes.
 *
 * @param n Number of nodes in the first set.
 * @param m Number of nodes in the second set.
 * @return Complete bipartite graph on `n + m` nodes and `n * m` undirected edges.
 */
template<typename... GraphArgs>
Graph complete_bipartite(const NodeID n, const NodeID m, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) { // set A
    builder.new_node();
    for (NodeID v = n; v < n + m; ++v) { builder.new_edge(v); }
  }
  for (NodeID u = n; u < n + m; ++u) { // set B
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) { builder.new_edge(v); }
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds the complete graph on `n` nodes.
 *
 * @param n Number of nodes in the graph.
 * @param edge_weight Weight used for all edges.
 * @return Complete graph with `n` nodes and `n * (n - 1)` undirected edges.
 */
template<typename... GraphArgs>
Graph complete(const NodeID n, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) {
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) {
      if (u != v) { builder.new_edge(v); }
    }
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds the star graph with `n` leaves, i.e., the complete bipartite graph on `(n, 1)` nodes. If nodes are weighted,
 * the center is the heaviest node with weight `2^(n + 1)`. *First node* is the center of the star.
 *
 * @param n Number of leaves.
 * @return Star graph with `n` leaves and one center.
 */
template<typename... GraphArgs>
Graph star(const NodeID n, GraphArgs &&...graph_args) {
  return complete_bipartite(1, n, std::forward<GraphArgs...>(graph_args)...);
}

template<typename... GraphArgs>
Graph matching(const NodeID m, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < 2 * m; u += 2) {
    builder.new_node();
    builder.new_edge(u + 1);
    builder.new_node();
    builder.new_edge(u);
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}
} // namespace graphs
} // namespace kaminpar::test