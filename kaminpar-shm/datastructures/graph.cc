/*******************************************************************************
 * Wrapper class that delegates all function calls to a concrete graph object.
 *
 * Most function calls are resolved via dynamic binding. Thus, they should not
 * be used when performance is critical. Instead, use an downcast and templatize
 * tight loops.
 *
 * @file:   graph.cc
 * @author: Daniel Seemaier
 * @date:   17.11.2023
 ******************************************************************************/
#include "kaminpar-shm/datastructures/graph.h"

#include <kassert/kassert.hpp>

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
Graph::Graph(std::unique_ptr<AbstractGraph> graph) : _underlying_graph(std::move(graph)) {}

//
// Utility debug functions
//

namespace debug {
void print_graph(const Graph &graph) {
  for (const NodeID u : graph.nodes()) {
    LLOG << "L" << u << " NW" << graph.node_weight(u) << " | ";
    for (const auto [e, v] : graph.neighbors(u)) {
      LLOG << "EW" << graph.edge_weight(e) << " L" << v << " NW" << graph.node_weight(v) << "  ";
    }
    LOG;
  }
}

bool validate_graph(
    const Graph &graph, const bool check_undirected, const NodeID num_pseudo_nodes
) {
  for (NodeID u = 0; u < graph.n(); ++u) {
    if (graph.raw_nodes()[u] > graph.raw_nodes()[u + 1]) {
      LOG_WARNING << "Bad node array at position " << u;
      return false;
    }
  }

  for (const NodeID u : graph.nodes()) {
    for (const auto [e, v] : graph.neighbors(u)) {
      if (v >= graph.n()) {
        LOG_WARNING << "Neighbor " << v << " of " << u << " is out-of-graph";
        return false;
      }
      if (u == v) {
        LOG_WARNING << "Self-loop at " << u;
        return false;
      }

      bool found_reverse = false;
      for (const auto [e_prime, u_prime] : graph.neighbors(v)) {
        if (u_prime >= graph.n()) {
          LOG_WARNING << "Neighbor " << u_prime << " of neighbor " << v << " of " << u
                      << " is out-of-graph";
          return false;
        }
        if (u != u_prime) {
          continue;
        }
        if (graph.edge_weight(e) != graph.edge_weight(e_prime)) {
          LOG_WARNING << "Weight of edge " << e << " (" << graph.edge_weight(e)
                      << ") differs from the weight of its reverse edge " << e_prime << " ("
                      << graph.edge_weight(e_prime) << ")";
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

Graph sort_neighbors(Graph graph) {
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

  Graph sorted_graph(std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  ));
  sorted_graph.set_permutation(graph.take_raw_permutation());

  return sorted_graph;
}
} // namespace debug
} // namespace kaminpar::shm
