/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/multi_refiner.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {
MultiRefiner::MultiRefiner(
    std::unordered_map<RefinementAlgorithm, std::unique_ptr<Refiner>> refiners,
    std::vector<RefinementAlgorithm> order
)
    : _refiners(std::move(refiners)),
      _order(std::move(order)) {}

void MultiRefiner::initialize(const PartitionedGraph &p_graph) {}

bool MultiRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  bool found_improvement = false;
  for (const RefinementAlgorithm algorithm : _order) {
    _refiners[algorithm]->initialize(p_graph);
    found_improvement |= _refiners[algorithm]->refine(p_graph, p_ctx);
  }
  return found_improvement;
}
} // namespace kaminpar::shm
