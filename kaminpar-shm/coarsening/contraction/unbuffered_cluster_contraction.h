/*******************************************************************************
 * @file:   unbuffered_cluster_contraction.h
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::contraction {
std::unique_ptr<CoarseGraph> contract_without_edgebuffer_remap(
    const Graph &graph,
    StaticArray<NodeID> &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
);
}

