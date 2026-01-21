#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class LHopRefiner : public Refiner {
public:
  LHopRefiner(const Context &ctx);

  LHopRefiner(const LHopRefiner &) = delete;
  LHopRefiner &operator=(const LHopRefiner &) = delete;

  LHopRefiner(LHopRefiner &&) noexcept = default;
  LHopRefiner &operator=(LHopRefiner &&) noexcept = delete;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  const Context &_ctx;

  const CSRGraph *_graph = nullptr;
};

} // namespace kaminpar::shm
