/*******************************************************************************
 * Basic independent set algorithm for distributed graphs.
 *
 * @file:   independent_set.h
 * @author: Daniel Seemaier
 * @date:   22.08.2022
 ******************************************************************************/
#pragma once

#include <vector>

#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/dkaminpar.h"

namespace kaminpar::dist::graph {
std::vector<NodeID>
find_independent_border_set(const DistributedPartitionedGraph &p_graph, int seed);
} // namespace kaminpar::dist::graph
