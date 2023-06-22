/*******************************************************************************
 * @file:   multi_refiner.h
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 * @brief:  Pseudo-refiner that runs multiple refiners in sequence.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class MultiRefiner : public Refiner {
public:
  MultiRefiner(std::vector<std::unique_ptr<Refiner>> refiners);

  MultiRefiner(const MultiRefiner &) = delete;
  MultiRefiner &operator=(const MultiRefiner &) = delete;
  MultiRefiner(MultiRefiner &&) noexcept = default;
  MultiRefiner &operator=(MultiRefiner &&) = delete;

  void initialize(const DistributedGraph &graph);
  void refine(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

private:
  std::vector<std::unique_ptr<Refiner>> _refiners;
};
} // namespace kaminpar::dist
