/*******************************************************************************
 * Pseudo-refiner that runs multiple refiners in sequence.
 *
 * @file:   multi_refiner.h
 * @author: Daniel Seemaier
 * @date:   08.08.2022
 ******************************************************************************/
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {
class MultiRefinerFactory : public GlobalRefinerFactory {
public:
  MultiRefinerFactory(
      std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefinerFactory>> factories,
      std::vector<RefinementAlgorithm> order
  );

  MultiRefinerFactory(const MultiRefinerFactory &) = delete;
  MultiRefinerFactory &operator=(const MultiRefinerFactory &) = delete;

  MultiRefinerFactory(MultiRefinerFactory &&) noexcept = default;
  MultiRefinerFactory &operator=(MultiRefinerFactory &&) = default;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefinerFactory>> _factories;
  std::vector<RefinementAlgorithm> _order;
};

class MultiRefiner : public GlobalRefiner {
public:
  MultiRefiner(
      std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefiner>> refiners,
      std::vector<RefinementAlgorithm> order
  );

  MultiRefiner(const MultiRefiner &) = delete;
  MultiRefiner &operator=(const MultiRefiner &) = delete;

  MultiRefiner(MultiRefiner &&) noexcept = default;
  MultiRefiner &operator=(MultiRefiner &&) = default;

  void initialize() final;
  bool refine() final;

private:
  std::unordered_map<RefinementAlgorithm, std::unique_ptr<GlobalRefiner>> _refiners;
  std::vector<RefinementAlgorithm> _order;
};
} // namespace kaminpar::dist
