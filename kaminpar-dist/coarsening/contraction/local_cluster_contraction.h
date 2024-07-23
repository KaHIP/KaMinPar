/*******************************************************************************
 * Graph contraction for local clusterings.
 *
 * @file:   local_cluster_contraction.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/contraction.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {
std::unique_ptr<CoarseGraph>
contract_local_clustering(const DistributedGraph &graph, const StaticArray<NodeID> &clustering);
} // namespace kaminpar::dist
