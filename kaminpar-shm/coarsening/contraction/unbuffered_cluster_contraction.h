/*******************************************************************************
 * @file:   unbuffered_cluster_contraction.h
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm::contraction {
std::unique_ptr<CoarseGraph> contract_without_edgebuffer_remap(
    const Graph &graph,
    scalable_vector<parallel::Atomic<NodeID>> &clustering,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
);
}

