/*******************************************************************************
 * @file:   noop_refiner.cc
 *
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Refiner that does nothing.
 ******************************************************************************/
#include "dkaminpar/refinement/noop_refiner.h"

namespace dkaminpar {
void NoopRefiner::initialize(const DistributedGraph &, const PartitionContext &) {}
void NoopRefiner::refine(DistributedPartitionedGraph &) {}
} // namespace dkaminpar
