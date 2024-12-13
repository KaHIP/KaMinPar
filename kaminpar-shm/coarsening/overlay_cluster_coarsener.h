/*******************************************************************************
 * Coarsener that computes multiple clusterings, overlays and contracts them to
 * coarsen the graph.
 *
 * @file:   overlay_cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   13.12.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class OverlayClusteringCoarsener : public Coarsener {
public:
  OverlayClusteringCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  OverlayClusteringCoarsener(const OverlayClusteringCoarsener &) = delete;
  OverlayClusteringCoarsener &operator=(const OverlayClusteringCoarsener) = delete;

  OverlayClusteringCoarsener(OverlayClusteringCoarsener &&) = delete;
  OverlayClusteringCoarsener &operator=(OverlayClusteringCoarsener &&) = delete;

  void initialize(const Graph *graph) final;

  bool coarsen() final;
  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final;

  [[nodiscard]] const Graph &current() const final {
    return _hierarchy.empty() ? *_input_graph : _hierarchy.back()->get();
  }

  [[nodiscard]] std::size_t level() const final {
    return _hierarchy.size();
  }

  void use_communities(std::span<const NodeID> communities) final;
  [[nodiscard]] std::span<const NodeID> current_communities() const final;

  void release_allocated_memory() final;

private:
  std::unique_ptr<CoarseGraph> pop_hierarchy(PartitionedGraph &&p_graph);

  [[nodiscard]] bool keep_allocated_memory() const;

  const Context &_ctx;
  const CoarseningContext &_c_ctx;
  const PartitionContext &_p_ctx;

  const Graph *_input_graph;
  std::vector<std::unique_ptr<CoarseGraph>> _hierarchy;

  std::span<const NodeID> _input_communities;
  std::vector<StaticArray<NodeID>> _communities_hierarchy;

  std::unique_ptr<Clusterer> _clustering_algorithm;

  contraction::MemoryContext _contraction_m_ctx{};
};

} // namespace kaminpar::shm

