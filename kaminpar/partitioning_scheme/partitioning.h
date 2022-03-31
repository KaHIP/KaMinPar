/*******************************************************************************
 * @file:   partitioning.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"

namespace kaminpar::partitioning {
PartitionedGraph partition(const Graph& graph, const Context& ctx);
}