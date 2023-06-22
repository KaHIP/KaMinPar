/*******************************************************************************
 * @file:   noop_refiner.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Refiner that does nothing.
 ******************************************************************************/
#include "dkaminpar/refinement/noop_refiner.h"

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
void NoopRefiner::initialize(const DistributedGraph &) {}
void NoopRefiner::refine(DistributedPartitionedGraph &, const PartitionContext &) {}
} // namespace kaminpar::dist
