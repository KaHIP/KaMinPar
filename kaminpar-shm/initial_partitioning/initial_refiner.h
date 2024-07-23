/*******************************************************************************
 * Interface for initial refinement algorithms.
 *
 * @file:   initial_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class InitialRefiner {
public:
  virtual ~InitialRefiner() = default;

  virtual void init(const CSRGraph &graph) = 0;
  virtual bool refine(PartitionedCSRGraph &p_graph, const PartitionContext &p_ctx) = 0;
};

std::unique_ptr<InitialRefiner> create_initial_refiner(const InitialRefinementContext &r_ctx);
} // namespace kaminpar::shm
