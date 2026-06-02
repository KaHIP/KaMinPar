/*******************************************************************************
 * Shared-memory implementation of JET, due to
 * "Jet: Multilevel Graph Partitioning on GPUs" by Gilbert et al.
 *
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

template <typename Graph> class JetRefinerImpl;

class JetRefiner : public Refiner {
public:
  JetRefiner(const Context &ctx);
  ~JetRefiner() override;

  JetRefiner(const JetRefiner &) = delete;
  JetRefiner &operator=(const JetRefiner &) = delete;

  JetRefiner(JetRefiner &&) noexcept = default;
  JetRefiner &operator=(JetRefiner &&) = delete;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
  std::unique_ptr<ReifiedGraphComponent<JetRefinerImpl>> _impl;
};

} // namespace kaminpar::shm
