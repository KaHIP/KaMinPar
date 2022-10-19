/*******************************************************************************
 * @file:   partitioning.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for partitioning schemes.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
DistributedPartitionedGraph partition(const DistributedGraph& graph, const Context& ctx);
} // namespace kaminpar::dist
