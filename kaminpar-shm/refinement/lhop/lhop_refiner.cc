#include "kaminpar-shm/refinement/lhop/lhop_refiner.h"

#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm {

LHopRefiner::LHopRefiner(const Context &ctx) : _ctx(ctx) {}

std::string LHopRefiner::name() const {
  return "L-hop";
}

void LHopRefiner::initialize(const PartitionedGraph &p_graph) {
  _graph = &concretize<CSRGraph>(p_graph.graph());
}

bool LHopRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  // Do nothing on coarse levels:
  if (p_graph.graph().level() > 0) {
    return false;
  }

  // <...>
  LOG << "Running toplevel L-hop refinement ...";

  // E.g.,
  // Move vertex 0 from block 1 to block 0, but only if the resulting weight of block 0 does not
  // exceed p_ctx.max_block_weight(0)
  const bool success = p_graph.move(0, 1, 0, p_ctx.max_block_weight(0));
  ((void)success);

  // Iterate over vertices (in parallel):
  _graph->pfor_nodes([&](const NodeID u) {
    // ... do stuff with u ...

    // Visit u's neighbors:
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight ew) {
      // Edge e, with weight ew, connects u to neighbor v
      ((void)e);
      ((void)v);
      ((void)ew);
    });
  });

  // Iterate over vertices sequentially:
  for (NodeID u = 0; u < _graph->n(); ++u) {
    // ... do stuff with u ...
  }

  // Indicate whether refinement improved the partition:
  // (mostly ignored)
  return false;
}

} // namespace kaminpar::shm
