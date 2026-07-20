/*******************************************************************************
 * Meta-refiner that refines an ensemble-contracted graph and the fine graph.
 *
 * @file:   meta_refiner.h
 ******************************************************************************/
#pragma once

#include <memory>
#include <span>

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/clustering/ensemble_clusterer.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class MetaRefiner final : public Refiner {
public:
  MetaRefiner(
      const Context &ctx, std::unique_ptr<Clusterer> lp_clusterer, std::unique_ptr<Refiner> refiner
  );

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;
  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  void set_communities(std::span<const NodeID> communities) final;

private:
  bool run_refiner(PartitionedGraph &p_graph, const PartitionContext &p_ctx);

  const Context &_ctx;
  EnsembleClusterer _ensemble_clusterer;
  std::unique_ptr<Refiner> _refiner;
  std::span<const NodeID> _communities;
};

} // namespace kaminpar::shm
