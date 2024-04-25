/*******************************************************************************
 * Contraction implementation that uses an edge buffer to store edges before
 * building the final graph.
 *
 * @file:   buffered_cluster_contraction.h
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::contraction {
std::unique_ptr<CoarseGraph> contract_clustering_buffered(
    const Graph &graph,
    StaticArray<NodeID> &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
);
}
