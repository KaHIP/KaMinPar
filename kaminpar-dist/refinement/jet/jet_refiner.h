/*******************************************************************************
 * Distributed JET refiner due to: "Jet: Multilevel Graph Partitioning on GPUs"
 * by Gilbert et al.
 *
 * @file:   jet_refiner.h
 * @author: Daniel Seemaier
 * @date:   02.05.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/refinement/refiner.h"

namespace kaminpar::dist {

class JetRefinerFactory : public GlobalRefinerFactory {
public:
  JetRefinerFactory(const Context &ctx);

  JetRefinerFactory(const JetRefinerFactory &) = delete;
  JetRefinerFactory &operator=(const JetRefinerFactory &) = delete;

  JetRefinerFactory(JetRefinerFactory &&) noexcept = default;
  JetRefinerFactory &operator=(JetRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

} // namespace kaminpar::dist
