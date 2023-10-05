/*******************************************************************************
 * Interface for refinement algorithms.
 *
 * @file:   refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::shm {
class Refiner {
public:
  Refiner(const Refiner &) = delete;
  Refiner &operator=(const Refiner &) = delete;

  Refiner(Refiner &&) noexcept = default;
  Refiner &operator=(Refiner &&) noexcept = default;

  virtual ~Refiner() = default;

  virtual void initialize(const PartitionedGraph &p_graph) = 0;

  virtual bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;

protected:
  Refiner() = default;
};

class NoopRefiner : public Refiner {
public:
  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &, const PartitionContext &) final {
    return false;
  }
};
} // namespace kaminpar::shm
