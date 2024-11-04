/*******************************************************************************
 * @file:   balanced_coarsener.h
 * @author: Daniel Seemaier
 * @date:   12.06.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class BalancedCoarsener : public Coarsener {
public:
  BalancedCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  BalancedCoarsener(const BalancedCoarsener &) = delete;
  BalancedCoarsener &operator=(const BalancedCoarsener) = delete;

  BalancedCoarsener(BalancedCoarsener &&) = delete;
  BalancedCoarsener &operator=(BalancedCoarsener &&) = delete;

  void initialize(const Graph *graph) final;

  bool coarsen() final;
  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final;

  [[nodiscard]] const Graph &current() const final {
    return _hierarchy.empty() ? *_input_graph : _hierarchy.back()->get();
  }

  [[nodiscard]] std::size_t level() const final {
    return _hierarchy.size();
  }

  void release_allocated_memory() final {}

private:
  std::unique_ptr<CoarseGraph> pop_hierarchy(PartitionedGraph &&p_graph);

  [[nodiscard]] bool keep_allocated_memory() const;

  const CoarseningContext &_c_ctx;
  const PartitionContext &_p_ctx;

  const Graph *_input_graph;
  std::vector<std::unique_ptr<CoarseGraph>> _hierarchy;

  std::unique_ptr<Clusterer> _clustering_algorithm;

  contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar::shm
