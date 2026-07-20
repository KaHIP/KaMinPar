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
#include <type_traits>

#include <tbb/parallel_for.h>

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
    const Graph &graph, StaticArray<NodeID> clustering, const ContractionCoarseningContext &con_ctx
) {
  MemoryContext m_ctx;
  return contract_clustering(graph, std::move(clustering), con_ctx, m_ctx);
}

std::unique_ptr<CoarseGraph> contract_clustering(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
) {
  switch (con_ctx.algorithm) {
  case ContractionAlgorithm::BUFFERED:
    return contract_clustering_buffered(graph, std::move(clustering), con_ctx, m_ctx);
  case ContractionAlgorithm::UNBUFFERED:
    return contract_clustering_unbuffered(graph, std::move(clustering), con_ctx, m_ctx);
  case ContractionAlgorithm::UNBUFFERED_NAIVE:
    return contract_clustering_unbuffered_naive(graph, std::move(clustering), con_ctx, m_ctx);
  }

  __builtin_unreachable();
}

void project_communities(
    CoarseGraph &coarse_graph,
    const std::span<const NodeID> fine_communities,
    const std::span<NodeID> coarse_communities
) {
  if constexpr (std::is_same_v<BlockID, NodeID>) {
    coarse_graph.project_down(
        {reinterpret_cast<const BlockID *>(fine_communities.data()), fine_communities.size()},
        {reinterpret_cast<BlockID *>(coarse_communities.data()), coarse_communities.size()}
    );
  } else {
    StaticArray<BlockID> fine(fine_communities.size());
    StaticArray<BlockID> coarse(coarse_communities.size());

    tbb::parallel_for<std::size_t>(0, fine.size(), [&](const std::size_t i) {
      fine[i] = static_cast<BlockID>(fine_communities[i]);
    });
    coarse_graph.project_down(fine, coarse);
    tbb::parallel_for<std::size_t>(0, coarse.size(), [&](const std::size_t i) {
      coarse_communities[i] = static_cast<NodeID>(coarse[i]);
    });
  }
}

} // namespace kaminpar::shm
