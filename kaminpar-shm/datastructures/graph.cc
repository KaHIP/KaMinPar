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

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

Graph::Graph(std::unique_ptr<AbstractGraph> graph) : _underlying_graph(std::move(graph)) {}

Graph::~Graph() = default;

Graph::Graph(Graph &&) noexcept = default;
Graph &Graph::operator=(Graph &&) noexcept = default;

NodeID Graph::n() const {
  return _underlying_graph->n();
}

EdgeID Graph::m() const {
  return _underlying_graph->m();
}

bool Graph::is_node_weighted() const {
  return _underlying_graph->is_node_weighted();
}

NodeWeight Graph::max_node_weight() const {
  return _underlying_graph->max_node_weight();
}

NodeWeight Graph::total_node_weight() const {
  return _underlying_graph->total_node_weight();
}

bool Graph::is_edge_weighted() const {
  return _underlying_graph->is_edge_weighted();
}

EdgeWeight Graph::total_edge_weight() const {
  return _underlying_graph->total_edge_weight();
}

bool Graph::sorted() const {
  return _underlying_graph->sorted();
}

void Graph::set_level(const int level) {
  _level = level;
}

int Graph::level() const {
  return _level;
}

AbstractGraph *Graph::underlying_graph() {
  return _underlying_graph.get();
}

const AbstractGraph *Graph::underlying_graph() const {
  return _underlying_graph.get();
}

CSRGraph &Graph::csr_graph() {
  return *dynamic_cast<CSRGraph *>(_underlying_graph.get());
}

const CSRGraph &Graph::csr_graph() const {
  return *dynamic_cast<const CSRGraph *>(_underlying_graph.get());
}

bool Graph::is_csr() const {
  return dynamic_cast<CSRGraph *>(_underlying_graph.get()) != nullptr;
}

CompressedGraph &Graph::compressed_graph() {
  return *dynamic_cast<CompressedGraph *>(_underlying_graph.get());
}

const CompressedGraph &Graph::compressed_graph() const {
  return *dynamic_cast<const CompressedGraph *>(_underlying_graph.get());
}

bool Graph::is_compressed() const {
  return dynamic_cast<CompressedGraph *>(_underlying_graph.get()) != nullptr;
}

namespace debug {

void print_graph(const Graph &graph) {
  reified(graph, [&](const auto &graph) {
    for (const NodeID u : graph.nodes()) {
      LLOG << "L" << u << " NW" << graph.node_weight(u) << " | ";
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        LLOG << "EW" << w << " L" << v << " NW" << graph.node_weight(v) << "  ";
      });
      LOG;
    }
  });
}

} // namespace debug

} // namespace kaminpar::shm
