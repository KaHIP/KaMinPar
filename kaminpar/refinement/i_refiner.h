/*******************************************************************************
 * @file:   i_refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"

namespace kaminpar::shm {
class IRefiner {
public:
  IRefiner(const IRefiner &) = delete;
  IRefiner &operator=(const IRefiner &) = delete;

  IRefiner(IRefiner &&) noexcept = default;
  IRefiner &operator=(IRefiner &&) noexcept = default;

  virtual ~IRefiner() = default;

  virtual void initialize(const Graph &graph) = 0;
  virtual bool refine(PartitionedGraph &p_graph,
                      const PartitionContext &p_ctx) = 0;
  [[nodiscard]] virtual EdgeWeight expected_total_gain() const = 0;

protected:
  IRefiner() = default;
};

class NoopRefiner : public IRefiner {
public:
  void initialize(const Graph &) final {}
  bool refine(PartitionedGraph &, const PartitionContext &) final {
    return false;
  }
  [[nodiscard]] EdgeWeight expected_total_gain() const final { return 0; }
};
} // namespace kaminpar::shm
