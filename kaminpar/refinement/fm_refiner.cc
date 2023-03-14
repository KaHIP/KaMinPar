/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#include "kaminpar/refinement/fm_refiner.h"

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"

namespace kaminpar::shm {
FMRefiner::FMRefiner(const Context &ctx) { ((void)ctx); }

void FMRefiner::initialize(const Graph &graph) { ((void)graph); }

bool FMRefiner::refine(PartitionedGraph &p_graph,
                       const PartitionContext &p_ctx) {
  ((void)p_graph);
  ((void)p_ctx);
  return false;
}

[[nodiscard]] EdgeWeight FMRefiner::expected_total_gain() const { return 0; }
} // namespace kaminpar::shm
