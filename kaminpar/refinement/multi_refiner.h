/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm {
class MultiRefiner : public Refiner {
public:
  MultiRefiner(std::vector<std::unique_ptr<Refiner>> refiners);

  MultiRefiner(const MultiRefiner &) = delete;
  MultiRefiner &operator=(const MultiRefiner &) = delete;

  MultiRefiner(MultiRefiner &&) = delete;
  MultiRefiner &operator=(MultiRefiner &&) = delete;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::vector<std::unique_ptr<Refiner>> _refiners;
};
} // namespace kaminpar::shm
