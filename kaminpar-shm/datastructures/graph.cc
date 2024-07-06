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

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {
Graph::Graph(std::unique_ptr<AbstractGraph> graph) : _underlying_graph(std::move(graph)) {}

//
// Utility debug functions
//

namespace debug {
void print_graph(const Graph &graph) {
  for (const NodeID u : graph.nodes()) {
    LLOG << "L" << u << " NW" << graph.node_weight(u) << " | ";
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      LLOG << "EW" << w << " L" << v << " NW" << graph.node_weight(v) << "  ";
    });
    LOG;
  }
}
} // namespace debug
} // namespace kaminpar::shm
