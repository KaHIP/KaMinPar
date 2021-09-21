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
