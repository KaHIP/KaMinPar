/*******************************************************************************
 * Pseudo-refiner that does nothing.
 *
 * @file:   noop_refiner.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#include "kaminpar-dist/refinement/noop_refiner.h"

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
std::unique_ptr<GlobalRefiner>
NoopRefinerFactory::create(DistributedPartitionedGraph &, const PartitionContext &) {
  return std::make_unique<NoopRefiner>();
}

void NoopRefiner::initialize() {}

bool NoopRefiner::refine() {
  return false;
}
} // namespace kaminpar::dist
