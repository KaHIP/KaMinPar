/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
/** @file */
#pragma once

#include "context.h"
#include "datastructure/graph.h"
#include "datastructure/marker.h"
#include "datastructure/queue.h"
#include "utility/random.h"

#include <utility>
#include <vector>

namespace kaminpar {
bool validate_graph(const Graph &graph);

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

std::pair<NodeID, NodeID> find_furthest_away_node(const Graph &graph, NodeID start_node, Queue<NodeID> &queue,
                                                  Marker<> &marker);

NodePermutations rearrange_and_remove_isolated_nodes(bool remove_isolated_nodes, PartitionContext &p_ctx,
                                                     StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
                                                     StaticArray<NodeWeight> &node_weights,
                                                     StaticArray<EdgeWeight> &edge_weights,
                                                     NodeWeight total_node_weight = -1);

PartitionedGraph revert_isolated_nodes_removal(PartitionedGraph p_graph, NodeID num_isolated_nodes,
                                               const PartitionContext &p_ctx);

/*!
 * Fast heuristic for finding two nodes with large distance: selects a random node (if seed_node is not specified),
 * performs a BFS and selects the last node processed as pseudo peripheral node. If the graph is disconnected, we select
 * a node in another connected component.
 *
 * @tparam seed_node If specified, start from this node instead of a random one (for unit tests).
 * @param graph
 * @param num_iterations Repeat the algorithm this many times for a chance of finding a pair of nodes with even larger
 * distance.
 * @return Pair of nodes with large distance between them.
 */
template<NodeID seed_node = kInvalidNodeID> // default: pick random nodes
std::pair<NodeID, NodeID> find_far_away_nodes(const Graph &graph, const std::size_t num_iterations = 1) {
  Queue<NodeID> queue(graph.n());
  Marker<> marker(graph.n());

  if constexpr (seed_node != kInvalidNodeID) { // for unit test
    return {seed_node, find_furthest_away_node(graph, seed_node, queue, marker).first};
  }

  NodeID best_distance = 0;
  std::pair<NodeID, NodeID> best_pair{0, 0};
  for (std::size_t i = 0; i < num_iterations; ++i) {
    const NodeID u = Randomize::instance().random_node(graph);
    const auto [v, distance] = find_furthest_away_node(graph, u, queue, marker);

    if (distance > best_distance || (distance == best_distance && Randomize::instance().random_bool())) {
      best_distance = distance;
      best_pair = {u, v};
    }
  }

  return best_pair;
}
} // namespace kaminpar
