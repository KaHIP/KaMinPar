/*******************************************************************************
 * @file:   partitioning.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for partitioning schemes.
 ******************************************************************************/

#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar {
DistributedPartitionedGraph partition(const DistributedGraph& graph, const Context& ctx);
} // namespace dkaminpar