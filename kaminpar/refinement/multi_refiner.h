/*******************************************************************************
 * @file:   multi_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Pseudo-refiner that runs multiple refiners in sequence.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/refinement/i_refiner.h"

namespace kaminpar::shm {
class MultiRefiner : public IRefiner {
public:
  MultiRefiner(std::vector<std::unique_ptr<IRefiner>> refiners);

  MultiRefiner(const MultiRefiner &) = delete;
  MultiRefiner &operator=(const MultiRefiner &) = delete;
  MultiRefiner(MultiRefiner &&) = delete;
  MultiRefiner &operator=(MultiRefiner &&) = delete;

  void initialize(const Graph &graph) final;
  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;
  [[nodiscard]] EdgeWeight expected_total_gain() const final;

private:
  std::vector<std::unique_ptr<IRefiner>> _refiners;
};
} // namespace kaminpar::shm
