/*******************************************************************************
 * Utility functions to find far-away nodes for BFS initialization.
 *
 * @file:   seed_node_utils.h
 * @author: Daniel Seemaier
 * @date:   21.09.21
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/queue.h"

namespace kaminpar::shm::ip {
/*!
 * Heuristic to find "far away" nodes for BFS initialization. Starts at a random seed
 * node and performs a BFS to find the furthest away node; repeats the process multiple
 * times for a higher chance at finding a good seed node.
 *
 * Allocates helper data structures with size O(n). Use the explicit version of this
 * function to avoid the extra allocations.
 *
 * @param graph the graph to search in.
 * @param num_iterations repeat the algorthm this many times for a chance of
 * finding a pair of nodes with even larger distance.
 *
 * @return Pair of hopefully far away nodes.
 */
std::pair<NodeID, NodeID> find_far_away_nodes(const CSRGraph &graph, int num_iterations);

/*!
 * Heuristic to find "far away" nodes for BFS initialization. Starts at a random seed
 * node and performs a BFS to find the furthest away node; repeats the process multiple
 * times for a higher chance at finding a good seed node.
 *
 * @param graph the graph to search in.
 * @param num_iterations repeat the algorthm this many times for a chance of
 * finding a pair of nodes with even larger distance.
 * @param queue a queue to use for BFS.
 * @param marker a marker to use for BFS.
 *
 * @return Pair of hopefully far away nodes.
 */
std::pair<NodeID, NodeID> find_far_away_nodes(
    const CSRGraph &graph, int num_iterations, Queue<NodeID> &queue, Marker<> &marker
);
} // namespace kaminpar::shm::ip
