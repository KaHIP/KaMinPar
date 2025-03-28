/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   threshold_sparsifying_cluster_coarsener.h
 * @author: Dominik Rosch, Daniel Seemaier
 * @date:   28.03.2025
 ******************************************************************************/
#pragma once

#include <memory>
#include <vector>

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class ThresholdSparsifyingClusteringCoarsener : public Coarsener {
public:
  ThresholdSparsifyingClusteringCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  ThresholdSparsifyingClusteringCoarsener(const ThresholdSparsifyingClusteringCoarsener &) = delete;
  ThresholdSparsifyingClusteringCoarsener &operator=(const ThresholdSparsifyingClusteringCoarsener
  ) = delete;

  ThresholdSparsifyingClusteringCoarsener(ThresholdSparsifyingClusteringCoarsener &&) = delete;
  ThresholdSparsifyingClusteringCoarsener &
  operator=(ThresholdSparsifyingClusteringCoarsener &&) = delete;

  void initialize(const Graph *graph) final;

  bool coarsen() final;

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final;

  void release_allocated_memory() override;

  [[nodiscard]] const Graph &current() const final {
    return _hierarchy.empty() ? *_input_graph : _hierarchy.back()->get();
  }

  [[nodiscard]] std::size_t level() const final {
    return _hierarchy.size();
  }

private:
  CSRGraph sparsify_and_recontract(CSRGraph csr, NodeID target_m) const;
  CSRGraph sparsify_and_make_negative_edges(CSRGraph csr, NodeID target_m) const;
  CSRGraph remove_negative_edges(CSRGraph csr) const;

  EdgeID sparsification_target(EdgeID old_m, NodeID old_n, EdgeID new_m) const;

  std::unique_ptr<CoarseGraph> pop_hierarchy(PartitionedGraph &&p_graph);

  [[nodiscard]] bool keep_allocated_memory() const;

  std::unique_ptr<Clusterer> _clustering_algorithm;
  const Context _ctx;
  const CoarseningContext &_c_ctx;
  const PartitionContext &_p_ctx;
  const SparsificationContext &_s_ctx;
  const Graph *_input_graph;
  std::vector<std::unique_ptr<CoarseGraph>> _hierarchy;
  contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar::shm
