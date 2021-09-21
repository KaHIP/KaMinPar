/*******************************************************************************
 * @file:   graph_permutation.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Computes graph permutations and builds the permuted graph.
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/datastructure/static_array.h"
#include "kaminpar/definitions.h"

namespace kaminpar::graph {
using NodePermutation = StaticArray<NodeID>;

struct NodePermutations {
  NodePermutation old_to_new;
  NodePermutation new_to_old;
};

NodePermutations sort_by_degree_buckets(const StaticArray<EdgeID> &nodes, bool deg0_position = false);

void build_permuted_graph(const StaticArray<EdgeID> &old_nodes, const StaticArray<NodeID> &old_edges,
                          const StaticArray<NodeWeight> &old_node_weights,
                          const StaticArray<EdgeWeight> &old_edge_weights, const NodePermutations &permutation,
                          StaticArray<EdgeID> &new_nodes, StaticArray<NodeID> &new_edges,
                          StaticArray<NodeWeight> &new_node_weights, StaticArray<EdgeWeight> &new_edge_weights);

std::pair<NodeID, NodeWeight> find_isolated_nodes_info(const StaticArray<EdgeID> &nodes,
                                                       const StaticArray<NodeWeight> &node_weights);

NodePermutations rearrange_and_remove_isolated_nodes(bool remove_isolated_nodes, PartitionContext &p_ctx,
                                                     StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
                                                     StaticArray<NodeWeight> &node_weights,
                                                     StaticArray<EdgeWeight> &edge_weights,
                                                     NodeWeight total_node_weight = -1);

PartitionedGraph revert_isolated_nodes_removal(PartitionedGraph p_graph, NodeID num_isolated_nodes,
                                               const PartitionContext &p_ctx);
} // namespace kaminpar::graph