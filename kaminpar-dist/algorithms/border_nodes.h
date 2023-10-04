/*******************************************************************************
 * Finds all border nodes of a partitioned graph.
 *
 * @file:   border_nodes.h
 * @author: Daniel Seemaier
 * @date:   20.09.2023
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::graph {
std::vector<NodeID> find_border_nodes(const DistributedPartitionedGraph &p_graph);
} // namespace kaminpar::dist::graph
