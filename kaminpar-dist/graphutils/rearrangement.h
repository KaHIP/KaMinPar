/*******************************************************************************
 * Sort and rearrange a graph by degree buckets.
 *
 * @file:   graph_rearrangement.h
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_csr_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist::graph {

DistributedCSRGraph rearrange(DistributedCSRGraph graph, const Context &ctx);

DistributedCSRGraph rearrange_by_degree_buckets(DistributedCSRGraph graph);

DistributedCSRGraph rearrange_by_coloring(DistributedCSRGraph graph, const Context &ctx);

DistributedCSRGraph rearrange_by_permutation(
    DistributedCSRGraph graph,
    StaticArray<NodeID> old_to_new,
    StaticArray<NodeID> new_to_old,
    bool degree_sorted
);

} // namespace kaminpar::dist::graph
