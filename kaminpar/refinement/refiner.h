/*******************************************************************************
 * @file:   refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Interface for refinement algorithms.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

namespace kaminpar::shm {
class Refiner {
public:
  Refiner(const Refiner &) = delete;
  Refiner &operator=(const Refiner &) = delete;

  Refiner(Refiner &&) noexcept = default;
  Refiner &operator=(Refiner &&) noexcept = default;

  virtual ~Refiner() = default;

  virtual void initialize(const PartitionedGraph &p_graph) = 0;

  virtual bool
  refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;

  [[nodiscard]] virtual EdgeWeight expected_total_gain() const = 0;

protected:
  Refiner() = default;
};

class NoopRefiner : public Refiner {
public:
  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &, const PartitionContext &) final {
    return false;
  }

  [[nodiscard]] EdgeWeight expected_total_gain() const final {
    return 0;
  }
};
} // namespace kaminpar::shm
