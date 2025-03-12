/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   cluster_coarsener.h
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#pragma once

#include "sparsification/Sampler.h"

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include <algorithm>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
class SparsifyingClusteringCoarsener : public Coarsener {
public:
  SparsifyingClusteringCoarsener(const Context &ctx, const PartitionContext &p_ctx);

  SparsifyingClusteringCoarsener(const SparsifyingClusteringCoarsener &) = delete;
  SparsifyingClusteringCoarsener &operator=(const SparsifyingClusteringCoarsener) = delete;

  SparsifyingClusteringCoarsener(SparsifyingClusteringCoarsener &&) = delete;
  SparsifyingClusteringCoarsener &operator=(SparsifyingClusteringCoarsener &&) = delete;

  void initialize(const Graph *graph) final;

  CSRGraph sparsify(const CSRGraph &csr, StaticArray<EdgeWeight> sample);
  EdgeID sparsificationTarget(EdgeID old_m, NodeID old_n, EdgeID new_m);

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
  std::unique_ptr<CoarseGraph> pop_hierarchy(PartitionedGraph &&p_graph);

  [[nodiscard]] bool keep_allocated_memory() const;

  const Context _ctx;
  const CoarseningContext &_c_ctx;
  const PartitionContext &_p_ctx;
  const SparsificationContext &_s_ctx;

  const Graph *_input_graph;
  std::vector<std::unique_ptr<CoarseGraph>> _hierarchy;

  std::unique_ptr<Clusterer> _clustering_algorithm;
  std::unique_ptr<sparsification::Sampler> _sampling_algorithm;

  contraction::MemoryContext _contraction_m_ctx{};
};
} // namespace kaminpar::shm
