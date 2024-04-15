/*******************************************************************************
 * @file:   naive_unbuffered_cluster_contraction.h
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::contraction {
std::unique_ptr<CoarseGraph> contract_clustering_unbuffered_naive(
    const Graph &graph,
    StaticArray<NodeID> &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
);
}
