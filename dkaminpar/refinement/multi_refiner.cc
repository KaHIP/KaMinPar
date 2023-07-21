/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 ******************************************************************************/
#include "dkaminpar/refinement/multi_refiner.h"

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
MultiRefinerFactory::MultiRefinerFactory(
    std::vector<std::unique_ptr<GlobalRefinerFactory>> factories
)
    : _factories(std::move(factories)) {}

std::unique_ptr<GlobalRefiner>
MultiRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  std::vector<std::unique_ptr<GlobalRefiner>> refiners;
  for (auto &factory : _factories) {
    refiners.push_back(factory->create(p_graph, p_ctx));
  }
  return std::make_unique<MultiRefiner>(std::move(refiners));
}

MultiRefiner::MultiRefiner(std::vector<std::unique_ptr<GlobalRefiner>> refiners)
    : _refiners(std::move(refiners)) {}

void MultiRefiner::initialize() {
  for (auto &refiner : _refiners) {
    refiner->initialize();
  }
}

bool MultiRefiner::refine() {
  bool improved_partition = false;
  for (auto &refiner : _refiners) {
    improved_partition |= refiner->refine();
  }
  return improved_partition;
}
} // namespace kaminpar::dist
