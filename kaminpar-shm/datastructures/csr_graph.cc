/*******************************************************************************
 * Static uncompressed CSR graph data structure.
 *
 * @file:   csr_graph.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {
template <template <typename> typename Container, template <typename> typename CompactContainer>
AbstractCSRGraph<Container, CompactContainer>::AbstractCSRGraph(const Graph &graph)
    : _nodes(graph.n() + 1),
      _edges(graph.m()),
      _node_weights(graph.n()),
      _edge_weights(graph.m()) {
  graph.reified([&](const auto &graph) {
    _nodes.front() = 0;
    graph.pfor_nodes([&](const NodeID u) {
      _nodes[u + 1] = graph.degree(u);
      _node_weights[u] = graph.node_weight(u);
    });
    parallel::prefix_sum(_nodes.begin(), _nodes.end(), _nodes.begin());

    graph.pfor_nodes([&](const NodeID u) {
      graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
        _edges[e] = v;
        _edge_weights[e] = graph.edge_weight(e);
      });
    });

    _total_node_weight = graph.total_node_weight();
    _total_edge_weight = graph.total_edge_weight();
    _max_degree = graph.max_degree();
    init_degree_buckets();
  });
}

template AbstractCSRGraph<StaticArray, StaticArray>::AbstractCSRGraph(const Graph &graph);

namespace debug {
bool validate_graph(
    const CSRGraph &graph, const bool check_undirected, const NodeID num_pseudo_nodes
) {
  for (NodeID u = 0; u < graph.n(); ++u) {
    if (graph.raw_nodes()[u] > graph.raw_nodes()[u + 1]) {
      LOG_WARNING << "Bad node array at position " << u;
      return false;
    }
  }

  for (const NodeID u : graph.nodes()) {
    for (EdgeID e = graph.first_edge(u); e < graph.first_invalid_edge(u); ++e) {
      if (e >= graph.m()) {
        LOG_WARNING << "Edge " << e << " of " << u << " is out-of-graph";
        return false;
      }

      const NodeID v = graph.edge_target(e);

      if (v >= graph.n()) {
        LOG_WARNING << "Neighbor " << v << " of " << u << " is out-of-graph";
        return false;
      }

      if (u == v) {
        LOG_WARNING << "Self-loop at " << u << ": " << e << " --> " << v;
        return false;
      }

      bool found_reverse = false;
      for (EdgeID e_prime = graph.first_edge(v); e_prime < graph.first_invalid_edge(v); ++e_prime) {
        if (e_prime >= graph.m()) {
          LOG_WARNING << "Edge " << e_prime << " of " << v << " is out-of-graph";
          std::exit(1);
          return false;
        }

        const NodeID u_prime = graph.edge_target(e_prime);

        if (u_prime >= graph.n()) {
          LOG_WARNING << "Neighbor " << u_prime << " of neighbor " << v << " of " << u
                      << " is out-of-graph";
          return false;
        }

        if (u != u_prime) {
          continue;
        }

        if (graph.edge_weight(e) != graph.edge_weight(e_prime)) {
          LOG_WARNING << "Weight of edge " << e << " (" << graph.edge_weight(e) << ") "   //
                      << "differs from the weight of its reverse edge " << e_prime << " " //
                      << "(" << graph.edge_weight(e_prime) << ")";                        //
          return false;
        }

        found_reverse = true;
        break;
      }

      if (check_undirected && v < graph.n() - num_pseudo_nodes && !found_reverse) {
        LOG_WARNING << "Edge " << u << " --> " << v << " exists with edge " << e
                    << ", but the reverse edges does not exist";
        return false;
      }
    }
  }

  return true;
}

CSRGraph sort_neighbors(CSRGraph graph) {
  const bool sorted = graph.sorted();
  const bool edge_weighted = graph.edge_weighted();

  StaticArray<EdgeID> nodes = graph.take_raw_nodes();
  StaticArray<NodeID> edges = graph.take_raw_edges();
  StaticArray<NodeWeight> node_weights = graph.take_raw_node_weights();
  StaticArray<EdgeWeight> edge_weights = graph.take_raw_edge_weights();

  if (edge_weighted) {
    StaticArray<std::pair<NodeID, EdgeWeight>> zipped(edges.size());
    tbb::parallel_for<EdgeID>(static_cast<EdgeID>(0), edges.size(), [&](const EdgeID e) {
      zipped[e] = {edges[e], edge_weights[e]};
    });

    tbb::parallel_for<NodeID>(0, nodes.size() - 1, [&](const NodeID u) {
      std::sort(
          zipped.begin() + nodes[u],
          zipped.begin() + nodes[u + 1],
          [](const auto &a, const auto &b) { return a.first < b.first; }
      );
    });

    tbb::parallel_for<EdgeID>(static_cast<EdgeID>(0), edges.size(), [&](const EdgeID e) {
      std::tie(edges[e], edge_weights[e]) = zipped[e];
    });
  } else {
    tbb::parallel_for<NodeID>(0, nodes.size() - 1, [&](const NodeID u) {
      std::sort(edges.begin() + nodes[u], edges.begin() + nodes[u + 1]);
    });
  }

  CSRGraph sorted_graph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );
  sorted_graph.set_permutation(graph.take_raw_permutation());

  return sorted_graph;
}
} // namespace debug
} // namespace kaminpar::shm
