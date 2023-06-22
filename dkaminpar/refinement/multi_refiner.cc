/*******************************************************************************
 * @file:   multi_refiner.cc
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 * @brief:  Pseudo-refiner that runs multiple refiners in sequence.
 ******************************************************************************/
#include "dkaminpar/refinement/multi_refiner.h"

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
MultiRefiner::MultiRefiner(std::vector<std::unique_ptr<Refiner>> refiners)
    : _refiners(std::move(refiners)) {}

void MultiRefiner::initialize(const DistributedGraph &graph) {
  for (const auto &refiner : _refiners) {
    refiner->initialize(graph);
  }
}

void MultiRefiner::refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  for (const auto &refiner : _refiners) {
    refiner->refine(p_graph, p_ctx);
  }
}
} // namespace kaminpar::dist
