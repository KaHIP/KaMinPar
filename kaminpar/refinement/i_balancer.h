/*******************************************************************************
 * @file:   i_balancer.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"

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