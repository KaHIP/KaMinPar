/*******************************************************************************
 * Contracts clusterings and constructs the coarse graph.
 *
 * @file:   cluster_contraction.cc
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"

#include <memory>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

// ... configurable contraction algorithms:
#include "kaminpar-shm/coarsening/contraction/buffered_cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/naive_unbuffered_cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/unbuffered_cluster_contraction.h"

namespace kaminpar::shm {

using namespace contraction;

std::unique_ptr<CoarseGraph> contract_clustering(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const ContractionCoarseningContext &con_ctx,
    [[maybe_unused]] Timer &timer
) {
  MemoryContext m_ctx;
  return contract_clustering(graph, std::move(clustering), con_ctx, m_ctx, timer);
}

std::unique_ptr<CoarseGraph> contract_clustering(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx,
    [[maybe_unused]] Timer &timer
) {
  switch (con_ctx.algorithm) {
  case ContractionAlgorithm::BUFFERED:
    return contract_clustering_buffered(graph, std::move(clustering), con_ctx, m_ctx, timer);
  case ContractionAlgorithm::UNBUFFERED:
    return contract_clustering_unbuffered(graph, std::move(clustering), con_ctx, m_ctx, timer);
  case ContractionAlgorithm::UNBUFFERED_NAIVE:
    return contract_clustering_unbuffered_naive(
        graph, std::move(clustering), con_ctx, m_ctx, timer
    );
  }

  __builtin_unreachable();
}

} // namespace kaminpar::shm
