/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   threshold_sparsifying_cluster_coarsener.h
 * @author: Dominik Rosch, Daniel Seemaier
 * @date:   28.03.2025
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/abstract_cluster_coarsener.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class ThresholdSparsifyingClusterCoarsener : public AbstractClusterCoarsener {
public:
  ThresholdSparsifyingClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  ThresholdSparsifyingClusterCoarsener(const ThresholdSparsifyingClusterCoarsener &) = delete;
  ThresholdSparsifyingClusterCoarsener &operator=(const ThresholdSparsifyingClusterCoarsener
  ) = delete;

  ThresholdSparsifyingClusterCoarsener(ThresholdSparsifyingClusterCoarsener &&) = delete;
  ThresholdSparsifyingClusterCoarsener &operator=(ThresholdSparsifyingClusterCoarsener &&) = delete;

  void use_communities(std::span<const NodeID> communities) final;

  bool coarsen() final;

private:
  CSRGraph sparsify_and_recontract(CSRGraph csr, NodeID target_m) const;
  CSRGraph sparsify_and_make_negative_edges(CSRGraph csr, NodeID target_m) const;
  CSRGraph remove_negative_edges(CSRGraph csr) const;

  EdgeID sparsification_target(EdgeID old_m, NodeID old_n, EdgeID new_m) const;

  const SparsificationContext &_s_ctx;
};

} // namespace kaminpar::shm
