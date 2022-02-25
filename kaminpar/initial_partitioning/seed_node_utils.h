/*******************************************************************************
 * @file:   seed_node_utils.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Algorithms to find seed nodes for initial partitioner based on
 * graph growing.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/datastructure/queue.h"

#include <utility>
#include <vector>

namespace kaminpar::ip {
std::pair<NodeID, NodeID> find_far_away_nodes(const Graph &graph, std::size_t num_iterations = 1);

std::pair<NodeID, NodeID> find_furthest_away_node(const Graph &graph, NodeID start_node, Queue<NodeID> &queue,
                                                  Marker<> &marker);
} // namespace kaminpar::ip
