#pragma once

#include "context.h"
#include "datastructure/graph.h"

namespace kaminpar {
class Refiner {
public:
  Refiner(const Refiner &) = delete;
  Refiner &operator=(const Refiner &) = delete;
  Refiner(Refiner &&) = delete;
  Refiner &operator=(Refiner &&) = delete;
  virtual ~Refiner() = default;

  virtual void initialize(const Graph &graph) = 0;
  virtual bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;
  virtual EdgeWeight expected_total_gain() const = 0;

protected:
  Refiner() = default;
};

class NoopRefiner : public Refiner {
public:
  void initialize(const Graph &) final {}
  bool refine(PartitionedGraph &, const PartitionContext &) final { return false; }
  EdgeWeight expected_total_gain() const final { return 0; }
};
} // namespace kaminpar