/*******************************************************************************
 * @file:   graph_rearrangement.h
 *
 * @author: Daniel Seemaier
 * @date:   17.11.2021
 * @brief:  Algorithms to rearrange graphs.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/graphutils/graph_permutation.h"

#include <utility>

namespace kaminpar::graph {
std::pair<NodeID, NodeWeight> find_isolated_nodes_info(const StaticArray<EdgeID> &nodes,
                                                       const StaticArray<NodeWeight> &node_weights);

NodePermutations<StaticArray> rearrange_graph(PartitionContext &p_ctx, StaticArray<EdgeID> &nodes,
                                              StaticArray<NodeID> &edges, StaticArray<NodeWeight> &node_weights,
                                              StaticArray<EdgeWeight> &edge_weights);

NodeID integrate_isolated_nodes(Graph &graph, double epsilon, Context &ctx);

PartitionedGraph assign_isolated_nodes(PartitionedGraph p_graph, const NodeID num_isolated_nodes,
                                       const PartitionContext &p_ctx);
} // namespace kaminpar::graph