/*******************************************************************************
 * @file:   kway_partitioning.h
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Multilevel graph partitioning using direct k-way initial
 * partitioning.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class KWayPartitioningScheme {
public:
    KWayPartitioningScheme(const DistributedGraph& graph, const Context& ctx);

    DistributedPartitionedGraph partition();

private:
    const DistributedGraph& _graph;
    const Context&          _ctx;
};
} // namespace kaminpar::dist
