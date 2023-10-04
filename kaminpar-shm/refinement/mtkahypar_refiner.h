/*******************************************************************************
 * Pseudo-refiner that calls Mt-KaHyPar.
 *
 * @file:   mtkahypar_refiner.cc
 * @author: Daniel Seemaier
 * @date:   01.07.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {
class MtKaHyParRefiner : public Refiner {
public:
  MtKaHyParRefiner(const Context &ctx);

  MtKaHyParRefiner(const MtKaHyParRefiner &) = delete;
  MtKaHyParRefiner &operator=(const MtKaHyParRefiner &) = delete;

  MtKaHyParRefiner(MtKaHyParRefiner &&) noexcept = default;
  MtKaHyParRefiner &operator=(MtKaHyParRefiner &&) = delete;

  void initialize(const PartitionedGraph &) {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx);

private:
  const Context &_ctx;
};
} // namespace kaminpar::shm
