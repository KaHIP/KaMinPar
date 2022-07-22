/*******************************************************************************
 * @file:   partitioning.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"

namespace kaminpar::shm::partitioning {
PartitionedGraph partition(const Graph& graph, const Context& ctx);
}
