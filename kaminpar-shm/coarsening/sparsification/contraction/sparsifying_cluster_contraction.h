/*******************************************************************************
 * @file:   unbuffered_cluster_contraction.h
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::contraction {

std::unique_ptr<CoarseGraph> contract_and_sparsify_clustering(
    const CSRGraph &graph,
    StaticArray<NodeID> mapping,
    NodeID c_n,
    EdgeWeight threshold_weight,
    double threshold_probability,
    const ContractionCoarseningContext &con_ctx,
    MemoryContext &m_ctx
);

}
