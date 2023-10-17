/*******************************************************************************
 * Adapter to use Mt-KaHyPar as a refinement algorithm.
 *
 * @file:   mtkahypar_refiner.h
 * @author: Daniel Seemaier
 * @date:   17.10.2023
 ******************************************************************************/
#pragma once 

#include "kaminpar-dist/refinement/refiner.h" 

namespace kaminpar::dist {
class MtKaHyParRefinerFactory : public GlobalRefinerFactory {
public:
  MtKaHyParRefinerFactory(const Context &ctx);

  MtKaHyParRefinerFactory(const MtKaHyParRefinerFactory &) = delete;
  MtKaHyParRefinerFactory &operator=(const MtKaHyParRefinerFactory &) = delete;

  MtKaHyParRefinerFactory(MtKaHyParRefinerFactory &&) noexcept = default;
  MtKaHyParRefinerFactory &operator=(MtKaHyParRefinerFactory &&) = delete;

  std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  const Context &_ctx;
};

class MtKaHyParRefiner : public GlobalRefiner {
  SET_STATISTICS_FROM_GLOBAL();
  SET_DEBUG(true);

public:
  MtKaHyParRefiner(
      const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
  );

  MtKaHyParRefiner(const MtKaHyParRefiner &) = delete;
  MtKaHyParRefiner &operator=(const MtKaHyParRefiner &) = delete;

  MtKaHyParRefiner(MtKaHyParRefiner &&) noexcept = default;
  MtKaHyParRefiner &operator=(MtKaHyParRefiner &&) = delete;

  void initialize() final;
  bool refine() final;

private:
  const Context &_ctx;
  DistributedPartitionedGraph &_p_graph;
  const PartitionContext &_p_ctx;
};
}
