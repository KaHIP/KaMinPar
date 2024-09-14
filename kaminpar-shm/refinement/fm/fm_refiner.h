/*******************************************************************************
 * Parallel k-way FM refinement algorithm.
 *
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class FMRefiner : public Refiner {
public:
  FMRefiner(const Context &ctx);

  // Note: requires dtor definition in the *.cc file due to the std::unique_ptr<> member.
  ~FMRefiner() override;

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) = delete;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;

  std::unique_ptr<Refiner> _core;
};

} // namespace kaminpar::shm
