#pragma once

#include "context.h"
#include "datastructure/graph.h"

namespace kaminpar {
class Balancer {
public:
  virtual ~Balancer() = default;
  virtual void initialize(const PartitionedGraph &p_graph) = 0;
  virtual bool balance(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;
};

class NoopBalancer : public Balancer {
public:
  void initialize(const PartitionedGraph &) final {}
  bool balance(PartitionedGraph &, const PartitionContext &) final { return true; }
};
} // namespace kaminpar