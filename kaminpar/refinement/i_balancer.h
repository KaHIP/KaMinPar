/*******************************************************************************
 * @file:   i_balancer.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"

namespace kaminpar::shm {
class IBalancer {
public:
  virtual ~IBalancer() = default;

  virtual void initialize(const PartitionedGraph &p_graph) = 0;
  virtual bool balance(PartitionedGraph &p_graph,
                       const PartitionContext &p_ctx) = 0;
};

class NoopBalancer : public IBalancer {
public:
  void initialize(const PartitionedGraph &) final {}
  bool balance(PartitionedGraph &, const PartitionContext &) final {
    return true;
  }
};
} // namespace kaminpar::shm
