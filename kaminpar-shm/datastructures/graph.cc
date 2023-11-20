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
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
Graph::Graph(std::unique_ptr<AbstractGraph> graph) : _underlying_graph(std::move(graph)) {}

//
// Utility debug functions
//

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

EdgeID compute_max_degree(const Graph &graph) {
  return parallel::max_difference(graph.raw_nodes().begin(), graph.raw_nodes().end());
}
} // namespace kaminpar::shm
