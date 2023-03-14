/*******************************************************************************
 * @file:   graph_rearrangement.h
 * @author: Daniel Seemaier
 * @date:   17.11.2021
 * @brief:  Algorithms to rearrange graphs.
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/graphutils/graph_permutation.h"

namespace kaminpar::shm::graph {
Graph rearrange_by_degree_buckets(Context &ctx, Graph graph);

NodePermutations<StaticArray>
rearrange_graph(PartitionContext &p_ctx, StaticArray<EdgeID> &nodes,
                StaticArray<NodeID> &edges,
                StaticArray<NodeWeight> &node_weights,
                StaticArray<EdgeWeight> &edge_weights);

NodeID integrate_isolated_nodes(Graph &graph, double epsilon, Context &ctx);

PartitionedGraph assign_isolated_nodes(PartitionedGraph p_graph,
                                       const NodeID num_isolated_nodes,
                                       const PartitionContext &p_ctx);
} // namespace kaminpar::shm::graph
