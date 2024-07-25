/*******************************************************************************
 * Initial refiner that does nothing.
 *
 * @file:   initial_noop_refiner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_noop_refiner.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
void InitialNoopRefiner::init(const CSRGraph &) {}

bool InitialNoopRefiner::refine(PartitionedCSRGraph &, const PartitionContext &) {
  return false;
}
} // namespace kaminpar::shm
