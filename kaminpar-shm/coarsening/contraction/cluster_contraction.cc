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

#include "kaminpar-shm/coarsening/contraction/buffered_cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/legacy_buffered_cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/naive_unbuffered_cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/unbuffered_cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm {
using namespace contraction;

std::unique_ptr<CoarseGraph> contract(
    const Graph &graph,
    const ContractionCoarseningContext &con_ctx,
    scalable_vector<parallel::Atomic<NodeID>> &clustering
) {
  MemoryContext m_ctx;
  return contract(graph, con_ctx, clustering, m_ctx);
}

std::unique_ptr<CoarseGraph> contract(
    const Graph &graph,
    const ContractionCoarseningContext &con_ctx,
    scalable_vector<parallel::Atomic<NodeID>> &clustering,
    MemoryContext &m_ctx
) {
  if (con_ctx.mode == ContractionMode::NO_EDGE_BUFFER_NAIVE) {
    return contract_without_edgebuffer_naive(graph, clustering, con_ctx, m_ctx);
  } else if (con_ctx.mode == ContractionMode::NO_EDGE_BUFFER_REMAP) {
    return contract_without_edgebuffer_remap(graph, clustering, con_ctx, m_ctx);
  } else if (con_ctx.mode == ContractionMode::EDGE_BUFFER_LEGACY) {
    return contract_with_edgebuffer_legacy(graph, clustering, con_ctx, m_ctx);
  } else {
    return contract_with_edgebuffer(graph, clustering, con_ctx, m_ctx);
  }
}
} // namespace kaminpar::shm
