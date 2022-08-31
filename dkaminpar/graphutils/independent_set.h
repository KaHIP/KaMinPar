/*******************************************************************************
 * @file:   independent_set.h
 * @author: Daniel Seemaier
 * @date:   22.08.2022
 * @brief:  Algorithm to find a independent set on distributed graphs.
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

namespace kaminpar::dist::graph {
tbb::concurrent_vector<NodeID> find_independent_border_set(const DistributedPartitionedGraph& p_graph, int seed);
} // namespace kaminpar::dist::graph