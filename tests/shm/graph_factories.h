#pragma once

#include <gmock/gmock.h>

#include "tests/shm/graph_builder.h"

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::testing {
/*!
 * Builds a graph with `n` nodes and zero edges.
 *
 * @param n Number of nodes in the graph.
 * @return Graph on `n` nodes and zero edges.
 */
template <typename... GraphArgs> Graph make_empty_graph(const NodeID n, GraphArgs &&...graph_args) {
  GraphBuilder builder(n, 0);
  for (NodeID u = 0; u < n; ++u) {
    builder.new_node();
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds a 2D grid graph.
 *
 * Edge nodes have degree 2, border nodes have degree 3, inner nodes have
 * degree 4. The 2D grid graph with parameters
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
template <typename... GraphArgs>
Graph make_grid_graph(const NodeID u, const NodeID v, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID i = 0; i < u; ++i) {
    const bool first_row = (i == 0);
    const bool last_row = (i + 1 == u);
    for (NodeID j = 0; j < v; ++j) {
      const bool first_column = (j == 0);
      const bool last_column = (j + 1 == v);
      const NodeID node = builder.new_node();
      if (!first_row) {
        builder.new_edge(node - v);
      }
      if (!last_row) {
        builder.new_edge(node + v);
      }
      if (!first_column) {
        builder.new_edge(node - 1);
      }
      if (!last_column) {
        builder.new_edge(node + 1);
      }
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
template <typename... GraphArgs>
Graph make_path_graph(const NodeID length, GraphArgs &&...graph_args) {
  return make_grid_graph(length, 1, std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds the complete bipartite graph on `n + m` nodes.
 *
 * @param n Number of nodes in the first set.
 * @param m Number of nodes in the second set.
 * @return Complete bipartite graph on `n + m` nodes and `n * m` undirected
 * edges.
 */
template <typename... GraphArgs>
Graph make_complete_bipartite_graph(const NodeID n, const NodeID m, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) { // set A
    builder.new_node();
    for (NodeID v = n; v < n + m; ++v) {
      builder.new_edge(v);
    }
  }
  for (NodeID u = n; u < n + m; ++u) { // set B
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) {
      builder.new_edge(v);
    }
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

template <typename Lambda, typename... GraphArgs>
Graph make_complete_bipartite_graph(
    const NodeID n, const NodeID m, Lambda &&l, GraphArgs &&...graph_args
) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) { // set A
    builder.new_node();
    for (NodeID v = n; v < n + m; ++v) {
      builder.new_edge(v, l(u, v));
    }
  }
  for (NodeID u = n; u < n + m; ++u) { // set B
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) {
      builder.new_edge(v, l(u, v));
    }
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
template <typename... GraphArgs>
Graph make_complete_graph(const NodeID n, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) {
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) {
      if (u != v) {
        builder.new_edge(v);
      }
    }
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

template <typename Lambda, typename... GraphArgs>
Graph make_complete_graph(const NodeID n, Lambda &&l, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < n; ++u) {
    builder.new_node();
    for (NodeID v = 0; v < n; ++v) {
      if (u != v) {
        builder.new_edge(v, l(u, v));
      }
    }
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}

/*!
 * Builds the star graph with `n` leaves, i.e., the complete bipartite graph on
 * `(n, 1)` nodes. If nodes are weighted, the center is the heaviest node with
 * weight `2^(n + 1)`. *First node* is the center of the star.
 *
 * @param n Number of leaves.
 * @return Star graph with `n` leaves and one center.
 */
template <typename... GraphArgs> Graph make_star_graph(const NodeID n, GraphArgs &&...graph_args) {
  return make_complete_bipartite_graph(1, n, std::forward<GraphArgs...>(graph_args)...);
}

template <typename Lambda, typename... GraphArgs>
Graph make_star_graph(const NodeID n, Lambda &&l, GraphArgs &&...graph_args) {
  return make_complete_bipartite_graph(
      1, n, std::forward<Lambda>(l), std::forward<GraphArgs...>(graph_args)...
  );
}

template <typename... GraphArgs>
Graph make_matching_graph(const NodeID m, GraphArgs &&...graph_args) {
  GraphBuilder builder;
  for (NodeID u = 0; u < 2 * m; u += 2) {
    builder.new_node();
    builder.new_edge(u + 1);
    builder.new_node();
    builder.new_edge(u);
  }
  return builder.build(std::forward<GraphArgs...>(graph_args)...);
}
} // namespace kaminpar::shm::testing
