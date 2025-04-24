/*******************************************************************************
 * Provides common functionality for coarseners optimized for cluster
 * contraction.
 *
 * @file:   abstract_cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   03.04.2025
 ******************************************************************************/
#pragma once

#include <span>
#include <vector>

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class AbstractClusterCoarsener : public Coarsener {
public:
  AbstractClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  void initialize(const Graph *graph) override;

  [[nodiscard]] const Graph &current() const override;

  [[nodiscard]] std::size_t level() const override;

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) override;

  void use_communities(std::span<const NodeID> communities) override;

  [[nodiscard]] std::span<const NodeID> current_communities() const override;

  void release_allocated_memory() override;

protected:
  void compute_clustering_for_current_graph(StaticArray<NodeID> &clustering);

  void contract_current_graph_and_push(StaticArray<NodeID> &clustering);

  [[nodiscard]] bool has_not_converged(NodeID prev_n) const;

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
