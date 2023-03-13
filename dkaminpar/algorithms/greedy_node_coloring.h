/***********************************************************************************************************************
 * @file:   greedy_node_coloring.h
 * @author: Daniel Seemaier
 * @date:   11.11.2022
 * @brief:  Distributed greedy node (vertex) coloring.
 **********************************************************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
using ColorID = EdgeID;

NoinitVector<ColorID>
compute_node_coloring_sequentially(const DistributedGraph &graph,
                                   NodeID number_of_supersteps);
} // namespace kaminpar::dist
