/*******************************************************************************
 * Debug utilities.
 *
 * @file:   debug.h
 * @author: Daniel Seemaier
 * @date:   31.03.2023
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist::debug {
void write_coarsest_graph(const DistributedGraph &graph, const DebugContext &d_ctx);

void write_metis_graph(const std::string &filename, const DistributedGraph &graph);

void write_coarsest_partition(
    const DistributedPartitionedGraph &p_graph, const DebugContext &d_ctx
);

void write_partition(
    const std::string &filename,
    const DistributedPartitionedGraph &p_graph,
    const bool use_original_node_order = true
);
} // namespace kaminpar::dist::debug
