#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/refiner.h"

namespace kaminpar::shm {
class MtKaHyParFMRefiner : public Refiner {
public:
  MtKaHyParFMRefiner(const Context &ctx);

  MtKaHyParFMRefiner(const MtKaHyParFMRefiner &) = delete;
  MtKaHyParFMRefiner &operator=(const MtKaHyParFMRefiner &) = delete;

  MtKaHyParFMRefiner(MtKaHyParFMRefiner &&) noexcept = default;
  MtKaHyParFMRefiner &operator=(MtKaHyParFMRefiner &&) = delete;

  void initialize(const PartitionedGraph &) {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx);

private:
  const Context &_ctx;
};
} // namespace kaminpar::shm

