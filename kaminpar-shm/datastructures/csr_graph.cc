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
  return validate_graph(
      graph.n(),
      graph.raw_nodes(),
      graph.raw_edges(),
      graph.raw_node_weights(),
      graph.raw_edge_weights(),
      check_undirected,
      num_pseudo_nodes
  );
}

bool validate_graph(
    const NodeID n,
    const StaticArray<EdgeID> &xadj,
    const StaticArray<NodeID> &adjncy,
    const StaticArray<NodeWeight> &vwgt,
    const StaticArray<EdgeWeight> &adjwgt,
    const bool check_undirected,
    const NodeID num_pseudo_nodes
) {
  if (xadj.size() < n + 1) {
    LOG_WARNING << "xadj is not large enough: " << xadj.size() << " < " << n + 1;
    return false;
  }

  if (!vwgt.empty() && vwgt.size() < n) {
    LOG_WARNING << "vwgt is not large enough: " << vwgt.size() << " < " << n;
    return false;
  }

  const EdgeID m = xadj[n];

  if (!adjwgt.empty() && adjwgt.size() < m) {
    LOG_WARNING << "adjwgt is not large enough: " << adjwgt.size() << " < " << xadj[n];
    return false;
  }

  for (NodeID u = 0; u < n; ++u) {
    if (xadj[u] > xadj[u + 1]) {
      LOG_WARNING << "Bad node array at position " << u;
      return false;
    }
  }

  for (NodeID u = 0; u < n; ++u) {
    for (EdgeID e = xadj[u]; e < xadj[u + 1]; ++e) {
      if (e >= m) {
        LOG_WARNING << "Edge " << e << " of " << u << " is out-of-graph";
        return false;
      }

      const NodeID v = adjncy[e];

      if (v >= n) {
        LOG_WARNING << "Neighbor " << v << " of " << u << " is out-of-graph";
        return false;
      }

      if (u == v) {
        LOG_WARNING << "Self-loop at " << u << ": " << e << " --> " << v;
        return false;
      }

      bool found_reverse = false;
      for (EdgeID e_prime = xadj[v]; e_prime < xadj[v + 1]; ++e_prime) {
        if (e_prime >= m) {
          LOG_WARNING << "Edge " << e_prime << " of " << v << " is out-of-graph";
          std::exit(1);
          return false;
        }

        const NodeID u_prime = adjncy[e_prime];

        if (u_prime >= n) {
          LOG_WARNING << "Neighbor " << u_prime << " of neighbor " << v << " of " << u
                      << " is out-of-graph";
          return false;
        }

        if (u != u_prime) {
          continue;
        }

        if (!adjwgt.empty() && adjwgt[e] != adjwgt[e_prime]) {
          LOG_WARNING << "Weight of edge " << e << " (" << adjwgt[e] << ") "              //
                      << "differs from the weight of its reverse edge " << e_prime << " " //
                      << "(" << adjwgt[e_prime] << ")";                                   //
          return false;
        }

        found_reverse = true;
        break;
      }

      if (check_undirected && v < n - num_pseudo_nodes && !found_reverse) {
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
