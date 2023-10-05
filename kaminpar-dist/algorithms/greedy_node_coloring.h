/*******************************************************************************
 * Basic implementation of a distributed vertex coloring algorithm.
 *
 * @file:   greedy_node_coloring.h
 * @author: Daniel Seemaier
 * @date:   11.11.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
using ColorID = EdgeID;

NoinitVector<ColorID>
compute_node_coloring_sequentially(const DistributedGraph &graph, NodeID number_of_supersteps);
} // namespace kaminpar::dist
