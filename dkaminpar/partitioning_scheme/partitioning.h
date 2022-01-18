/*******************************************************************************
 * @file:   partitioning.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for partitioning schemes.
 ******************************************************************************/

#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/context.h"

namespace dkaminpar {
DistributedPartitionedGraph partition(const DistributedGraph &graph, const Context &ctx);
} // namespace dkaminpar