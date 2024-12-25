/*******************************************************************************
 * Stores edges from ghost nodes to local nodes of the graph from which the ghost
 * graph is constructed.
 *
 * @file:   ghost_graph.cc
 * @author: Daniel Seemaier
 * @date:   25.12.2024
 ******************************************************************************/
#include "kaminpar-dist/datastructures/ghost_graph.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::dist {

namespace {

template <typename Graph>
void construct(
    const Graph &graph,
    StaticArray<EdgeID> &xadj,
    StaticArray<NodeID> &adjncy,
    StaticArray<EdgeWeight> &adjwgt
) {
  xadj.resize(graph.ghost_n() + 1);

  graph.pfor_nodes([&](const NodeID u) {
    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (graph.is_ghost_node(v)) {
        const NodeID nth_ghost = v - graph.n();
        __atomic_fetch_add(&xadj[nth_ghost], 1, __ATOMIC_RELAXED);
      }
    });
  });

  parallel::prefix_sum(xadj.begin(), xadj.end(), xadj.begin());
  adjncy.resize(xadj.back());

  const bool has_edge_weights = graph.is_edge_weighted();
  if (has_edge_weights) {
    adjwgt.resize(xadj.back());
  }

  graph.pfor_nodes([&](const NodeID u) {
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (graph.is_ghost_node(v)) {
        const NodeID nth_ghost = v - graph.n();
        const EdgeID pos = __atomic_sub_fetch(&xadj[nth_ghost], 1, __ATOMIC_RELAXED);
        adjncy[pos] = u;
        if (has_edge_weights) {
          adjwgt[pos] = weight;
        }
      }
    });
  });
}

} // namespace

GhostGraph::GhostGraph() = default;

GhostGraph::GhostGraph(const DistributedGraph &graph) {
  initialize(graph);
}

void GhostGraph::initialize(const DistributedGraph &graph) {
  graph.reified([&](const auto &graph) { construct(graph, _xadj, _adjncy, _adjwgt); });
  _n = graph.n();
  _ghost_n = graph.ghost_n();
}

} // namespace kaminpar::dist
