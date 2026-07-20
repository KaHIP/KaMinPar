/*******************************************************************************
 * Coarsener that computes multiple clusterings, overlays and contracts them to
 * coarsen the graph.
 *
 * @file:   overlay_cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   13.12.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/abstract_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/clustering/ensemble_clusterer.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class OverlayClusterCoarsener : public AbstractClusterCoarsener {
public:
  OverlayClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  OverlayClusterCoarsener(const OverlayClusterCoarsener &) = delete;
  OverlayClusterCoarsener &operator=(const OverlayClusterCoarsener) = delete;

  OverlayClusterCoarsener(OverlayClusterCoarsener &&) = delete;
  OverlayClusterCoarsener &operator=(OverlayClusterCoarsener &&) = delete;

  bool coarsen() final;

private:
  EnsembleClusterer *_ensemble_clusterer = nullptr;
};

} // namespace kaminpar::shm
