/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 ******************************************************************************/
#include "kaminpar-dist/refinement/multi_refiner.h"

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {
MultiRefinerFactory::MultiRefinerFactory(
    std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefinerFactory>> factories,
    std::vector<RefinementAlgorithm> order
)
    : _factories(std::move(factories)),
      _order(std::move(order)) {}

std::unique_ptr<GlobalRefiner>
MultiRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefiner>> refiners;
  for (const auto &[algorithm, factory] : _factories) {
    refiners[algorithm] = factory->create(p_graph, p_ctx);
  }
  return std::make_unique<MultiRefiner>(std::move(refiners), _order);
}

MultiRefiner::MultiRefiner(
    std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefiner>> refiners,
    std::vector<RefinementAlgorithm> order
)
    : _refiners(std::move(refiners)),
      _order(std::move(order)) {}

void MultiRefiner::initialize() {}

bool MultiRefiner::refine() {
  bool improved_partition = false;
  for (const RefinementAlgorithm algorithm : _order) {
    _refiners[algorithm]->initialize();
    improved_partition |= _refiners[algorithm]->refine();
  }
  return improved_partition;
}
} // namespace kaminpar::dist
