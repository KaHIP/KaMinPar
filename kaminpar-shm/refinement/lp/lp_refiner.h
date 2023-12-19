/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.h
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

template <typename Graph> class LabelPropagationRefinerImpl;

class LabelPropagationRefiner : public Refiner {
public:
  LabelPropagationRefiner(const Context &ctx);

  ~LabelPropagationRefiner() override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  std::unique_ptr<LabelPropagationRefinerImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LabelPropagationRefinerImpl<CompressedGraph>> _compressed_impl;
};
} // namespace kaminpar::shm
