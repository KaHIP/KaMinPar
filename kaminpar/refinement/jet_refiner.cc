#include "kaminpar/refinement/jet_refiner.h"

#include "common/noinit_vector.h"

namespace kaminpar::shm {
JetRefiner::JetRefiner(const Context &ctx) : _ctx(ctx) {}

bool JetRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {

  ((void)p_graph);
  ((void)p_ctx);
  return false;
}
} // namespace kaminpar::shm
