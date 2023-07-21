/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.h
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 ******************************************************************************/
#pragma once

#include <vector>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class MultiRefinerFactory : public GlobalRefinerFactory {
public:
  MultiRefinerFactory(std::vector<std::unique_ptr<GlobalRefinerFactory>> factories);

  MultiRefinerFactory(const MultiRefinerFactory &) = delete;
  MultiRefinerFactory &operator=(const MultiRefinerFactory &) = delete;

  MultiRefinerFactory(MultiRefinerFactory &&) noexcept = default;
  MultiRefinerFactory &operator=(MultiRefinerFactory &&) = default;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::vector<std::unique_ptr<GlobalRefinerFactory>> _factories;
};

class MultiRefiner : public GlobalRefiner {
public:
  MultiRefiner(std::vector<std::unique_ptr<GlobalRefiner>> refiners);

  MultiRefiner(const MultiRefiner &) = delete;
  MultiRefiner &operator=(const MultiRefiner &) = delete;

  MultiRefiner(MultiRefiner &&) noexcept = default;
  MultiRefiner &operator=(MultiRefiner &&) = default;

  void initialize() final;
  bool refine() final;

private:
  std::vector<std::unique_ptr<GlobalRefiner>> _refiners;
};
} // namespace kaminpar::dist
