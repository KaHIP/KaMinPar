#pragma once

#include "context.h"
#include "datastructure/graph.h"

namespace kaminpar::partitioning {
PartitionedGraph partition(const Graph &graph, const Context &ctx);
}
