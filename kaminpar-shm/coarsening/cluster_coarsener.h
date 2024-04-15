/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
class ClusteringCoarsener : public Coarsener {
public:
  ClusteringCoarsener(
      std::unique_ptr<Clusterer> clustering_algorithm, const CoarseningContext &c_ctx
  )
      : _clustering_algorithm(std::move(clustering_algorithm)),
        _c_ctx(c_ctx) {}

  ClusteringCoarsener(const ClusteringCoarsener &) = delete;
  ClusteringCoarsener &operator=(const ClusteringCoarsener) = delete;

  ClusteringCoarsener(ClusteringCoarsener &&) = delete;
  ClusteringCoarsener &operator=(ClusteringCoarsener &&) = delete;

  void initialize(const Graph *graph) final;

  bool
  coarsen(NodeWeight max_cluster_weight, NodeID to_size, const bool free_memory_afterwards) final;

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final;

  [[nodiscard]] const Graph &current() const final {
    return _hierarchy.empty() ? *_input_graph : _hierarchy.back()->get();
  }

  [[nodiscard]] std::size_t level() const final {
    return _hierarchy.size();
  }

private:
  const CoarseningContext &_c_ctx;

  const Graph *_input_graph;
  std::vector<std::unique_ptr<CoarseGraph>> _hierarchy;

  StaticArray<NodeID> _clustering{};
  std::unique_ptr<Clusterer> _clustering_algorithm;

  contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar::shm
