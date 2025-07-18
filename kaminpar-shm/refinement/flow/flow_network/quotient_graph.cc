#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

void QuotientGraph::reconstruct() {
  SCOPED_TIMER("Construct Quotient Graph");
  const CSRGraph &graph = _p_graph.graph();

  for (Edge &edge : _edges) {
    edge.cut_edges.clear();
    edge.cut_weight = 0;
  }

  EdgeWeight total_cut_weight = 0;
  for (const NodeID u : graph.nodes()) {
    const BlockID u_block = _p_graph.block(u);

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      const BlockID v_block = _p_graph.block(v);
      if (u_block >= v_block) {
        return;
      }

      Edge &quotient_edge = edge(u_block, v_block);
      quotient_edge.cut_edges.emplace_back(u, v);
      quotient_edge.cut_weight += w;

      total_cut_weight += w;
    });
  }

  Random &random = Random::instance();
  for (BlockID block2 = 0, k = _p_graph.k(); block2 < k; ++block2) {
    for (BlockID block1 = 0; block1 < block2; ++block1) {
      Edge &quotient_edge = edge(block1, block2);
      random.shuffle(quotient_edge.cut_edges);
    }
  }

  KASSERT(
      metrics::edge_cut_seq(_p_graph) == total_cut_weight,
      "Computed an invalid total cut weight",
      assert::heavy
  );

  _total_cut_weight = total_cut_weight;
}

void QuotientGraph::add_gain(
    const BlockID b1,
    const BlockID b2,
    const EdgeWeight gain,
    std::span<const GraphEdge> new_cut_edges
) {
  KASSERT(b1 != b2);

  Edge &quotient_edge = edge(b1, b2);
  quotient_edge.total_gain += gain;

  for (const GraphEdge &cut_edge : new_cut_edges) {
    const NodeID u = cut_edge.u;
    const NodeID v = cut_edge.v;

    const BlockID u_block = _p_graph.block(u);
    const BlockID v_block = _p_graph.block(v);

    const bool edge_order = u_block < v_block;
    edge(u_block, v_block).cut_edges.emplace_back(edge_order ? u : v, edge_order ? v : u);
  }

  _total_cut_weight -= gain;
}

} // namespace kaminpar::shm
