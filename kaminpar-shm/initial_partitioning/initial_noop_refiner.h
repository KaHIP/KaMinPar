/*******************************************************************************
 * Initial refiner that does nothing.
 *
 * @file:   initial_noop_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class InitialNoopRefiner : public InitialRefiner {
public:
  void init(const CSRGraph &graph) final;

  bool refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) final;
};
} // namespace kaminpar::shm
