/*******************************************************************************
 * @file:   i_refiner.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"

namespace kaminpar {
class IRefiner {
public:
  IRefiner(const IRefiner &) = delete;
  IRefiner &operator=(const IRefiner &) = delete;
  IRefiner(IRefiner &&) = delete;
  IRefiner &operator=(IRefiner &&) = delete;

  virtual ~IRefiner() = default;

  virtual void initialize(const Graph &graph) = 0;
  virtual bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;
  [[nodiscard]] virtual EdgeWeight expected_total_gain() const = 0;

protected:
  IRefiner() = default;
};

class NoopRefiner : public IRefiner {
public:
  void initialize(const Graph &) final {}
  bool refine(PartitionedGraph &, const PartitionContext &) final { return false; }
  [[nodiscard]] EdgeWeight expected_total_gain() const final { return 0; }
};
} // namespace kaminpar