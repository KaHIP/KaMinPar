/*******************************************************************************
 * Shared-memory implementation of JET, due to
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 *
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {
class JetRefiner : public Refiner {
public:
  JetRefiner(const Context &ctx);

  JetRefiner(const JetRefiner &) = delete;
  JetRefiner &operator=(const JetRefiner &) = delete;

  JetRefiner(JetRefiner &&) noexcept = default;
  JetRefiner &operator=(JetRefiner &&) = delete;

  void initialize(const PartitionedGraph &) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;

  double _negative_gain_factor;
};
} // namespace kaminpar::shm
