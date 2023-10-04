/*******************************************************************************
 * @file:   seed_node_utils.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Algorithms to find seed nodes for initial partitioner based on
 * graph growing.
 ******************************************************************************/
#pragma once

#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/queue.h"

namespace kaminpar::shm::ip {
/*!
 * Fast heuristic for finding two nodes with large distance: selects a random
 * node (if seed_node is not specified), performs a BFS and selects the last
 * node processed as pseudo peripheral node. If the graph is disconnected, we
 * select a node in another connected component.
 *
 * @tparam seed_node If specified, start from this node instead of a random one
 * (for unit tests).
 * @param graph
 * @param num_iterations Repeat the graphutils this many times for a chance of
 * finding a pair of nodes with even larger distance.
 * @return Pair of nodes with large distance between them.
 */
std::pair<NodeID, NodeID> find_far_away_nodes(const Graph &graph, std::size_t num_iterations = 1);

std::pair<NodeID, NodeID> find_furthest_away_node(
    const Graph &graph, NodeID start_node, Queue<NodeID> &queue, Marker<> &marker
);
} // namespace kaminpar::shm::ip
