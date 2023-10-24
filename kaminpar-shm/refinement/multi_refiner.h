/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {
class MultiRefiner : public Refiner {
public:
  MultiRefiner(
      std::unordered_map<RefinementAlgorithm, std::unique_ptr<Refiner>> refiners,
      std::vector<RefinementAlgorithm> order
  );

  MultiRefiner(const MultiRefiner &) = delete;
  MultiRefiner &operator=(const MultiRefiner &) = delete;

  MultiRefiner(MultiRefiner &&) = delete;
  MultiRefiner &operator=(MultiRefiner &&) = delete;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::unordered_map<RefinementAlgorithm, std::unique_ptr<Refiner>> _refiners;
  std::vector<RefinementAlgorithm> _order;
};
} // namespace kaminpar::shm
