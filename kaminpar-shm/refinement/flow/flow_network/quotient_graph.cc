#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

QuotientGraph::QuotientGraph(const PartitionedCSRGraph &p_graph)
    : _p_graph(p_graph),
      _edges(p_graph.k() * p_graph.k()) {
  for (Edge &edge : _edges) {
    edge.cut_weight = 0;
    edge.total_gain = 0;
  }

  reconstruct();
}

void QuotientGraph::reconstruct() {
  SCOPED_TIMER("Construct Quotient Graph");

  for (Edge &edge : _edges) {
    edge.cut_edges.clear();
    edge.cut_weight = 0;
  }

  const CSRGraph &graph = _p_graph.graph();

  EdgeWeight total_cut_weight = 0;
  for (const NodeID u : graph.nodes()) {
    const BlockID u_block = _p_graph.block(u);

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      const BlockID v_block = _p_graph.block(v);

      if (u_block < v_block) {
        Edge &quotient_edge = edge(u_block, v_block);
        quotient_edge.cut_edges.emplace_back(u, v);
        quotient_edge.cut_weight += w;

        total_cut_weight += w;
      }
    });
  }

  _total_cut_weight = total_cut_weight;
  KASSERT(
      metrics::edge_cut_seq(_p_graph) == total_cut_weight,
      "Computed an invalid total cut weight",
      assert::heavy
  );
}

void QuotientGraph::add_gain(const BlockID b1, const BlockID b2, const EdgeWeight gain) {
  Edge &quotient_edge = edge(b1, b2);
  quotient_edge.total_gain += gain;

  __atomic_fetch_sub(&_total_cut_weight, gain, __ATOMIC_RELAXED);
}

void QuotientGraph::add_cut_edges(std::span<const GraphEdge> new_cut_edges) {
  SCOPED_TIMER("Update Quotient Graph");

  for (const GraphEdge &cut_edge : new_cut_edges) {
    const NodeID u = cut_edge.u;
    const NodeID v = cut_edge.v;

    const BlockID u_block = _p_graph.block(u);
    const BlockID v_block = _p_graph.block(v);
    if (u_block == v_block) {
      continue;
    }

    const bool u_comes_first = u_block < v_block;
    edge(u_block, v_block).cut_edges.emplace_back(u_comes_first ? u : v, u_comes_first ? v : u);
  }
}

} // namespace kaminpar::shm
