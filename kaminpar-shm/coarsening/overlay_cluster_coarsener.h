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
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

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
  StaticArray<NodeID> overlay(StaticArray<NodeID> a, const StaticArray<NodeID> &b);
};

} // namespace kaminpar::shm
