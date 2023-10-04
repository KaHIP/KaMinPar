/*******************************************************************************
 * Sort and rearrange a graph by degree buckets.
 *
 * @file:   graph_rearrangement.h
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::dist::graph {
DistributedGraph rearrange(DistributedGraph graph, const Context &ctx);

DistributedGraph rearrange_by_degree_buckets(DistributedGraph graph);

DistributedGraph rearrange_by_coloring(DistributedGraph graph, const Context &ctx);

DistributedGraph rearrange_by_permutation(
    DistributedGraph graph,
    StaticArray<NodeID> old_to_new,
    StaticArray<NodeID> new_to_old,
    bool degree_sorted
);
} // namespace kaminpar::dist::graph
