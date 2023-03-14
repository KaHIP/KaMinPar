/*******************************************************************************
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Pseudo-refiner that runs multiple refiners in sequence.
 ******************************************************************************/
#include "kaminpar/refinement/multi_refiner.h"

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/refinement/i_refiner.h"

namespace kaminpar::shm {
MultiRefiner::MultiRefiner(std::vector<std::unique_ptr<IRefiner>> refiners)
    : _refiners(std::move(refiners)) {}

void MultiRefiner::initialize(const Graph &graph) {
  for (const auto &refiner : _refiners) {
    refiner->initialize(graph);
  }
}

bool MultiRefiner::refine(PartitionedGraph &p_graph,
                          const PartitionContext &p_ctx) {
  bool found_improvement = false;
  for (const auto &refiner : _refiners) {
    found_improvement |= refiner->refine(p_graph, p_ctx);
  }
  return found_improvement;
}

EdgeWeight MultiRefiner::expected_total_gain() const {
  EdgeWeight expected_gain = 0;
  for (const auto &refiner : _refiners) {
    expected_gain += refiner->expected_total_gain();
  }
  return expected_gain;
}
} // namespace kaminpar::shm
